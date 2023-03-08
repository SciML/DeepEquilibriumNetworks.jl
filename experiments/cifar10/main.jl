import CUDA, DEQExperiments, FluxMPI, Logging, Lux, OneHotArrays, Optimisers, PyCall,
       Random, Setfield, SimpleConfig, Statistics, Wandb
import Lux.Training
import ComponentArrays as CA

# Dataloaders
function get_dataloaders(; augment, data_root, eval_batchsize, train_batchsize)
  tf = PyCall.pyimport("tensorflow")
  tfds = PyCall.pyimport("tensorflow_datasets")

  tf.config.set_visible_devices([], "GPU")

  ds_train, ds_test = tfds.load("cifar10"; split=["train", "test"], as_supervised=true,
                                data_dir=data_root)

  image_mean = tf.constant([[[0.4914f0, 0.4822f0, 0.4465f0]]])
  image_std = tf.constant([[[0.2023f0, 0.1994f0, 0.2010f0]]])

  function normalize(img, label)
    img = tf.cast(img, tf.float32) / 255.0f0
    img = (img - image_mean) / image_std
    return img, label
  end

  ds_train = ds_train.cache()
  ds_test = ds_test.cache().map(normalize)
  if augment
    tf_rng = tf.random.Generator.from_seed(12345; alg="philox")
    function augmentation(img, label)
      seed = tf_rng.make_seeds(2)[1]

      img, label = normalize(img, label)
      img = tf.image.stateless_random_flip_left_right(img, seed)
      img = tf.image.resize(img, (42, 42))
      img = tf.image.stateless_random_crop(img, (32, 32, 3), seed)

      return img, label
    end
    ds_train = ds_train.map(augmentation; num_parallel_calls=tf.data.AUTOTUNE)
  else
    ds_train = ds_train.map(normalize; num_parallel_calls=tf.data.AUTOTUNE)
  end

  if DEQExperiments.is_distributed()
    ds_train = ds_train.shard(FluxMPI.total_worders(), FluxMPI.local_rank())
    ds_test = ds_test.shard(FluxMPI.total_worders(), FluxMPI.local_rank())
  end

  ds_train = ds_train.prefetch(tf.data.AUTOTUNE).shuffle(1024).repeat(-1)
  ds_test = ds_test.prefetch(tf.data.AUTOTUNE).repeat(1)

  return (tfds.as_numpy(ds_train.batch(train_batchsize)),
          tfds.as_numpy(ds_test.batch(eval_batchsize)))
end

function _data_postprocess(image, label)
  return (Lux.gpu(permutedims(image, (3, 2, 4, 1))),
          Lux.gpu(OneHotArrays.onehotbatch(label, 0:9)))
end

function main(filename, args)
  cfg = SimpleConfig.define_configuration(args, DEQExperiments.ExperimentConfig, filename)

  return main(splitext(basename(filename))[1], cfg)
end

function main(config_name::String, cfg::DEQExperiments.ExperimentConfig)
  rng = Random.Xoshiro()
  Random.seed!(rng, cfg.seed)

  model = DEQExperiments.construct(cfg.model)

  loss_function = DEQExperiments.get_loss_function(cfg)

  opt, sched = DEQExperiments.construct(cfg.optimizer)

  tstate = if cfg.model.model_type != "neural_ode"
    Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
  else
    ps, st = Lux.setup(rng, model)
    ps = ps |> CA.ComponentArray |> Lux.gpu
    st = st |> Lux.gpu
    opt_state = Optimisers.setup(opt, ps)
    Training.TrainState(model, ps, st, opt_state, 0)
  end
  vjp_rule = Training.ZygoteVJP()

  DEQExperiments.warmup_model(loss_function, model, tstate.parameters, tstate.states, cfg;
                              transform_input=Lux.gpu)

  ds_train, ds_test = get_dataloaders(; cfg.dataset.augment, cfg.dataset.data_root,
                                      cfg.dataset.eval_batchsize,
                                      cfg.dataset.train_batchsize)
  _, ds_train_iter = iterate(ds_train)

  # Setup
  expt_name = ("config-$(config_name)_continuous-$(cfg.model.solver.continuous)" *
               "_type-$(cfg.model.model_type)_seed-$(cfg.seed)" *
               "_jfb-$(cfg.model.sensealg.jfb)_id-$(cfg.train.expt_id)")

  ckpt_dir = joinpath(cfg.train.expt_subdir, cfg.train.checkpoint_dir, expt_name)
  log_dir = joinpath(cfg.train.expt_subdir, cfg.train.log_dir, expt_name)
  if cfg.train.resume == ""
    rpath = joinpath(ckpt_dir, "model_current.jlso")
  else
    rpath = cfg.train.resume
  end

  ckpt = DEQExperiments.load_checkpoint(rpath)
  if !isnothing(ckpt)
    tstate = ckpt.tstate
    initial_step = ckpt.step
    DEQExperiments.should_log() && @info "Training Started from Step: $initial_step"
  else
    initial_step = 1
  end

  if cfg.train.pretrain_steps != 0
    if DEQExperiments.should_log()
      @info "Will pretrain for $(cfg.train.pretrain_steps) steps"
    end
    Setfield.@set! tstate.states = Lux.update_state(tstate.states, :fixed_depth, Val(5))
  end

  # Setup Logging
  loggers = DEQExperiments.create_logger(log_dir, cfg.train.total_steps - initial_step,
                                         cfg.train.total_steps - initial_step, expt_name,
                                         SimpleConfig.flatten_configuration(cfg))

  best_test_accuracy = 0

  for step in initial_step:(cfg.train.total_steps)
    # Train Step
    t = time()
    (x, y), ds_train_iter = iterate(ds_train, ds_train_iter)
    x, y = _data_postprocess(x, y)
    data_time = time() - t

    bsize = size(x, ndims(x))

    ret_val = DEQExperiments.run_training_step(vjp_rule, loss_function, (x, y), tstate)
    loss, _, stats, tstate, gs, step_stats = ret_val
    Setfield.@set! tstate.states = Lux.update_state(tstate.states, :update_mask, Val(true))

    # LR Update
    lr_new = sched(step + 1)
    Setfield.@set! tstate.optimizer_state = Optimisers.adjust(tstate.optimizer_state,
                                                              lr_new)

    accuracy = DEQExperiments.accuracy(Lux.cpu(stats.y_pred), Lux.cpu(y))
    residual = abs(Statistics.mean(stats.residual))

    # Logging
    loggers.progress_loggers.train.avg_meters.batch_time(data_time +
                                                         step_stats.fwd_time +
                                                         step_stats.bwd_time +
                                                         step_stats.opt_time, bsize)
    loggers.progress_loggers.train.avg_meters.data_time(data_time, bsize)
    loggers.progress_loggers.train.avg_meters.fwd_time(step_stats.fwd_time, bsize)
    loggers.progress_loggers.train.avg_meters.bwd_time(step_stats.bwd_time, bsize)
    loggers.progress_loggers.train.avg_meters.opt_time(step_stats.opt_time, bsize)
    loggers.progress_loggers.train.avg_meters.loss(loss, bsize)
    loggers.progress_loggers.train.avg_meters.ce_loss(stats.ce_loss, bsize)
    loggers.progress_loggers.train.avg_meters.skip_loss(stats.skip_loss, bsize)
    loggers.progress_loggers.train.avg_meters.residual(residual, bsize)
    loggers.progress_loggers.train.avg_meters.top1(accuracy, bsize)
    loggers.progress_loggers.train.avg_meters.top5(-1, bsize)
    loggers.progress_loggers.train.avg_meters.nfe(stats.nfe, bsize)

    if step % cfg.train.print_frequency == 1 && DEQExperiments.should_log()
      DEQExperiments.print_meter(loggers.progress_loggers.train.progress, step)
      log_vals = DEQExperiments.get_loggable_values(loggers.progress_loggers.train.progress)
      loggers.csv_loggers.train(step, log_vals...)
      Wandb.log(loggers.wandb_logger, loggers.log_functions.train(step, log_vals...))
      DEQExperiments.reset_meter!(loggers.progress_loggers.train.progress)
    end

    if step == cfg.train.pretrain_steps
      DEQExperiments.should_log() && @info "Pretraining Completed!!!"
      Setfield.@set! tstate.states = Lux.update_state(tstate.states, :fixed_depth, Val(0))
    end

    # Free memory eagarly
    CUDA.unsafe_free!(x)
    CUDA.unsafe_free!(y)

    if step % cfg.train.evaluate_every == 1 || step == cfg.train.total_steps
      is_best = true

      st_eval = Lux.testmode(tstate.states)
      for (x, y) in ds_test
        t = time()
        x, y = _data_postprocess(x, y)
        dtime = time() - t

        t = time()
        loss, st_, stats = loss_function(model, tstate.parameters, st_eval, (x, y))
        fwd_time = time() - t

        bsize = size(x, ndims(x))

        acc = DEQExperiments.accuracy(Lux.cpu(stats.y_pred), Lux.cpu(y))

        loggers.progress_loggers.eval.avg_meters.batch_time(dtime + fwd_time, bsize)
        loggers.progress_loggers.eval.avg_meters.data_time(dtime, bsize)
        loggers.progress_loggers.eval.avg_meters.fwd_time(fwd_time, bsize)
        loggers.progress_loggers.eval.avg_meters.loss(loss, bsize)
        loggers.progress_loggers.eval.avg_meters.ce_loss(stats.ce_loss, bsize)
        loggers.progress_loggers.eval.avg_meters.skip_loss(stats.skip_loss, bsize)
        loggers.progress_loggers.eval.avg_meters.residual(abs(Statistics.mean(stats.residual)),
                                                          bsize)
        loggers.progress_loggers.eval.avg_meters.top1(acc, bsize)
        loggers.progress_loggers.eval.avg_meters.top5(-1, bsize)
        loggers.progress_loggers.eval.avg_meters.nfe(stats.nfe, bsize)

        # Free memory eagarly
        CUDA.unsafe_free!(x)
        CUDA.unsafe_free!(y)
      end

      if DEQExperiments.should_log()
        DEQExperiments.print_meter(loggers.progress_loggers.eval.progress, step)
        log_vals = DEQExperiments.get_loggable_values(loggers.progress_loggers.eval.progress)
        loggers.csv_loggers.eval(step, log_vals...)
        Wandb.log(loggers.wandb_logger, loggers.log_functions.eval(step, log_vals...))
        DEQExperiments.reset_meter!(loggers.progress_loggers.eval.progress)
      end

      accuracy = loggers.progress_loggers.eval.avg_meters.top1.average
      is_best = accuracy >= best_test_accuracy
      if is_best
        best_test_accuracy = accuracy
      end

      ckpt = (tstate=tstate, step=initial_step)
      DEQExperiments.save_checkpoint(ckpt; is_best,
                                     filename=joinpath(ckpt_dir, "model_$(step).jlso"))
    end
  end

  return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
  main(ARGS[1], ARGS[2:end])
end
