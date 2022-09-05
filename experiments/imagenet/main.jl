import Augmentor, CUDA, DEQExperiments, FluxMPI, Images, Logging, Lux, MLUtils,
       OneHotArrays, Optimisers, Random, Setfield, SimpleConfig, Statistics, Wandb
import Lux.Training

FluxMPI.Init(; verbose=true)

# Dataloaders
struct ImageDataset
  image_files::Any
  labels::Any
  mapping::Any
  augmentation_pipeline::Any
  normalization_parameters::Any
end

function ImageDataset(folder::String, augmentation_pipeline, normalization_parameters)
  ulabels = readdir(folder)
  label_dirs = joinpath.((folder,), ulabels)
  @assert length(label_dirs)==1000 "There should be 1000 subdirectories in $folder"

  classes = readlines(joinpath(@__DIR__, "synsets.txt"))
  mapping = Dict(z => i for (i, z) in enumerate(ulabels))

  istrain = endswith(folder, r"train|train/")

  if istrain
    image_files = vcat(map((x, y) -> joinpath.((x,), y), label_dirs,
                           readdir.(label_dirs))...)

    remove_files = [
      "n01739381_1309.JPEG",
      "n02077923_14822.JPEG",
      "n02447366_23489.JPEG",
      "n02492035_15739.JPEG",
      "n02747177_10752.JPEG",
      "n03018349_4028.JPEG",
      "n03062245_4620.JPEG",
      "n03347037_9675.JPEG",
      "n03467068_12171.JPEG",
      "n03529860_11437.JPEG",
      "n03544143_17228.JPEG",
      "n03633091_5218.JPEG",
      "n03710637_5125.JPEG",
      "n03961711_5286.JPEG",
      "n04033995_2932.JPEG",
      "n04258138_17003.JPEG",
      "n04264628_27969.JPEG",
      "n04336792_7448.JPEG",
      "n04371774_5854.JPEG",
      "n04596742_4225.JPEG",
      "n07583066_647.JPEG",
      "n13037406_4650.JPEG",
      "n02105855_2933.JPEG",
    ]
    remove_files = joinpath.((folder,),
                             joinpath.(first.(rsplit.(remove_files, "_", limit=2)),
                                       remove_files))

    image_files = [setdiff(Set(image_files), Set(remove_files))...]

    labels = [mapping[x] for x in map(x -> x[2], rsplit.(image_files, "/", limit=3))]
  else
    vallist = hcat(split.(readlines(joinpath(@__DIR__, "val_list.txt")))...)
    labels = parse.(Int, vallist[2, :]) .+ 1
    filenames = [joinpath(classes[l], vallist[1, i]) for (i, l) in enumerate(labels)]
    image_files = joinpath.((folder,), filenames)
    idxs = findall(isfile, image_files)
    image_files = image_files[idxs]
    labels = labels[idxs]
  end

  return ImageDataset(image_files, labels, mapping, augmentation_pipeline,
                      normalization_parameters)
end

function Base.getindex(data::ImageDataset, i::Int)
  img = Images.load(data.image_files[i])
  img = Augmentor.augment(img, data.augmentation_pipeline)
  cimg = Images.channelview(img)
  if ndims(cimg) == 2
    cimg = reshape(cimg, 1, size(cimg, 1), size(cimg, 2))
    cimg = vcat(cimg, cimg, cimg)
  end
  img = Float32.(permutedims(cimg, (3, 2, 1)))
  img = (img .- data.normalization_parameters.mean) ./ data.normalization_parameters.std
  return img, OneHotArrays.onehot(data.labels[i], 1:1000)
end

Base.length(data::ImageDataset) = length(data.image_files)

function get_dataloaders(; augment, data_root, eval_batchsize, train_batchsize)
  normalization_parameters = (mean=reshape([0.485f0, 0.456f0, 0.406f0], 1, 1, 3),
                              std=reshape([0.229f0, 0.224f0, 0.225f0], 1, 1, 3))
  train_data_augmentation = Augmentor.Resize(256, 256) |>
                            Augmentor.FlipX(0.5) |>
                            Augmentor.RCropSize(224, 224)
  val_data_augmentation = Augmentor.Resize(256, 256) |> Augmentor.CropSize(224, 224)
  train_dataset = ImageDataset(joinpath(data_root, "train"), train_data_augmentation,
                               normalization_parameters)
  val_dataset = ImageDataset(joinpath(data_root, "val"), val_data_augmentation,
                             normalization_parameters)
  if DEQExperiments.is_distributed()
    train_dataset = FluxMPI.DistributedDataContainer(train_dataset)
    val_dataset = FluxMPI.DistributedDataContainer(val_dataset)
  end

  train_data = MLUtils.BatchView(MLUtils.shuffleobs(train_dataset);
                                 batchsize=train_batchsize, partial=false, collate=true)

  val_data = MLUtils.BatchView(val_dataset; batchsize=eval_batchsize, partial=true,
                               collate=true)

  train_iter = Iterators.cycle(MLUtils.eachobsparallel(train_data;
                                                       executor=FLoops.ThreadedEx(),
                                                       buffer=true))

  val_iter = MLUtils.eachobsparallel(val_data; executor=FLoops.ThreadedEx(), buffer=true)

  return train_iter, val_iter
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

  ds_train, ds_val = get_dataloaders(; cfg.dataset.augment, cfg.dataset.data_root,
                                     cfg.dataset.eval_batchsize,
                                     cfg.dataset.train_batchsize)
  _, ds_train_iter = iterate(ds_train)

  tstate = Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
  tstate = DEQExperiments.is_distributed() ? FluxMPI.synchronize!(tstate; root_rank=0) :
           tstate
  vjp_rule = Training.ZygoteVJP()

  DEQExperiments.warmup_model(loss_function, model, tstate.parameters, tstate.states, cfg;
                              transform_input=Lux.gpu)

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

  best_val_accuracy = 0

  for step in initial_step:(cfg.train.total_steps)
    # Train Step
    t = time()
    (x, y), ds_train_iter = iterate(ds_train, ds_train_iter)
    x = x |> Lux.gpu
    y = y |> Lux.gpu
    data_time = time() - t

    bsize = size(x, ndims(x))

    ret_val = DEQExperiments.run_training_step(vjp_rule, loss_function, (x, y), tstate)
    loss, _, stats, tstate, gs, step_stats = ret_val
    Setfield.@set! tstate.states = Lux.update_state(tstate.states, :update_mask, Val(true))

    # LR Update
    lr_new = sched(step + 1)
    Setfield.@set! tstate.optimizer_state = Optimisers.adjust(tstate.optimizer_state,
                                                              lr_new)

    acc1, acc5 = DEQExperiments.accuracy(Lux.cpu(stats.y_pred), Lux.cpu(y), (1, 5))
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
    loggers.progress_loggers.train.avg_meters.top1(acc1, bsize)
    loggers.progress_loggers.train.avg_meters.top5(acc5, bsize)
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

    if step % cfg.train.evaluate_every == 1 || step == cfg.train.total_steps
      is_best = true

      st_eval = Lux.testmode(tstate.states)
      for (x, y) in ds_val
        t = time()
        x = x |> Lux.gpu
        y = y |> Lux.gpu
        dtime = time() - t

        t = time()
        loss, st_, stats = loss_function(model, tstate.parameters, st_eval, (x, y))
        fwd_time = time() - t

        bsize = size(x, ndims(x))

        acc1, acc5 = DEQExperiments.accuracy(Lux.cpu(stats.y_pred), Lux.cpu(y), (1, 5))

        loggers.progress_loggers.eval.avg_meters.batch_time(dtime + fwd_time, bsize)
        loggers.progress_loggers.eval.avg_meters.data_time(dtime, bsize)
        loggers.progress_loggers.eval.avg_meters.fwd_time(fwd_time, bsize)
        loggers.progress_loggers.eval.avg_meters.loss(loss, bsize)
        loggers.progress_loggers.eval.avg_meters.ce_loss(stats.ce_loss, bsize)
        loggers.progress_loggers.eval.avg_meters.skip_loss(stats.skip_loss, bsize)
        loggers.progress_loggers.eval.avg_meters.residual(abs(Statistics.mean(stats.residual)),
                                                          bsize)
        loggers.progress_loggers.eval.avg_meters.top1(acc1, bsize)
        loggers.progress_loggers.eval.avg_meters.top5(acc5, bsize)
        loggers.progress_loggers.eval.avg_meters.nfe(stats.nfe, bsize)
      end

      if DEQExperiments.should_log()
        DEQExperiments.print_meter(loggers.progress_loggers.eval.progress, step)
        log_vals = DEQExperiments.get_loggable_values(loggers.progress_loggers.eval.progress)
        loggers.csv_loggers.eval(step, log_vals...)
        Wandb.log(loggers.wandb_logger, loggers.log_functions.eval(step, log_vals...))
        DEQExperiments.reset_meter!(loggers.progress_loggers.eval.progress)
      end

      accuracy = loggers.progress_loggers.eval.avg_meters.top1.average
      is_best = accuracy >= best_val_accuracy
      if is_best
        best_val_accuracy = accuracy
      end

      ckpt = (tstate=tstate, step=initial_step)
      if DEQExperiments.should_log()
        DEQExperiments.save_checkpoint(ckpt; is_best,
                                       filename=joinpath(ckpt_dir, "model_$(step).jlso"))
      end
    end
  end

  return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
  main(ARGS[1], ARGS[2:end])
end
