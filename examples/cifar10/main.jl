# Adapted from https://github.com/avik-pal/Lux.jl/tree/main/examples/ImageNet/main.jl

using ArgParse                                          # Parse Arguments from Commandline
using DataAugmentation                                  # Image Augmentation
using CUDA                                              # GPUs <3
using DataLoaders                                       # Pytorch like DataLoaders
using Dates                                             # Printing current time
using FastDEQ                                           # Deep Equilibrium Model
using FastDEQExperiments                                # Models built using FastDEQ
using FluxMPI                                           # Distibuted Training
using Formatting                                        # Pretty Printing
using Functors                                          # Parameter Manipulation
using Images                                            # Image Processing
using LinearAlgebra                                     # Linear Algebra
using Lux                                               # Neural Network Framework
using MLDataPattern                                     # Data Pattern
using MLDatasets                                        # CIFAR10
using MLDataUtils                                       # Shuffling and Splitting Data
using MLUtils                                           # Data Processing
using NNlib                                             # Neural Network Backend
using OneHotArrays                                      # One Hot Encoding
using Optimisers                                        # Collection of Gradient Based Optimisers
using ParameterSchedulers                               # Collection of Schedulers for Parameter Updates
using Random                                            # Make things less Random
using Serialization                                     # Serialize Models
using Setfield                                          # Easy Parameter Manipulation
using Statistics                                        # Statistics
using ValueHistories                                    # Storing Value Histories
using Zygote                                            # Our AD Engine

# Distributed Training
FluxMPI.Init(; verbose=true)
CUDA.allowscalar(false)

# Training Options
include("options.jl")

function get_experiment_config(args)
    return get_experiment_configuration(
        Val(:CIFAR10),
        Val(Symbol(args["model-size"]));
        model_type=Symbol(args["model-type"]),
        continuous=!args["discrete"],
        abstol=args["abstol"],
        reltol=args["reltol"],
        jfb=args["jfb"],
        train_batchsize=args["train-batchsize"],
        eval_batchsize=args["eval-batchsize"],
        seed=args["seed"],
        w_skip=args["w-skip"],
    )
end

create_model(expt_config, args) = get_model(expt_config; device=gpu, warmup=true, loss_function=get_loss_function(args))

function get_loss_function(args)
    if args["model-type"] == "VANILLA"
        function loss_function_closure_vanilla(x, y, model, ps, st)
            (ŷ, soln), st_ = model(x, ps, st)
            celoss = logitcrossentropy(ŷ, y)
            skiploss = FastDEQExperiments.mae(soln.u₀, soln.z_star)
            loss = celoss
            return loss, st_, (ŷ, soln.nfe, celoss, skiploss, soln.residual)
        end
        return loss_function_closure_vanilla
    else
        function loss_function_closure_skip(x, y, model, ps, st)
            (ŷ, soln), st_ = model(x, ps, st)
            celoss = logitcrossentropy(ŷ, y)
            skiploss = FastDEQExperiments.mae(soln.u₀, soln.z_star)
            loss = celoss + args["w-skip"] * skiploss
            return loss, st_, (ŷ, soln.nfe, celoss, skiploss, soln.residual)
        end
        return loss_function_closure_skip
    end
end

# Checkpointing
function save_checkpoint(state, is_best, filename)
    if should_log()
        isdir(dirname(filename)) || mkpath(dirname(filename))
        serialize(filename, state)
        is_best && cp(filename, joinpath(dirname(filename), "model_best.jls"); force=true)
    end
end

# DataLoading
struct CIFARDataContainer
    images
    labels
    transform
end

function get_dataloaders(expt_config::NamedTuple)
    x_train, y_train = CIFAR10.traindata(Float32)
    x_test, y_test = CIFAR10.testdata(Float32)

    x_train_images = map(x -> Image(colorview(RGB, permutedims(x, (3, 2, 1)))), eachslice(x_train; dims=4))
    y_train = collect(eachslice(Float32.(onehotbatch(y_train, 0:9)); dims=2))

    x_test_images = map(x -> Image(colorview(RGB, permutedims(x, (3, 2, 1)))), eachslice(x_test; dims=4))
    y_test = collect(eachslice(Float32.(onehotbatch(y_test, 0:9)); dims=2))

    base_transform = ImageToTensor() |> Normalize((0.4914f0, 0.4822f0, 0.4465f0), (0.2023f0, 0.1994f0, 0.2010f0))

    if expt_config.augment
        train_transform = ScaleKeepAspect((36, 36)) |> RandomResizeCrop((32, 32)) |> Maybe(FlipX()) |> base_transform
    else
        train_transform = base_transform
    end

    train_dataset = MLUtils.shuffleobs(CIFARDataContainer(x_train_images, y_train, train_transform))
    train_dataset = is_distributed() ? DistributedDataContainer(train_dataset) : train_dataset
    test_dataset = CIFARDataContainer(x_test_images, y_test, base_transform)
    test_dataset = is_distributed() ? DistributedDataContainer(test_dataset) : test_dataset

    return (
        DataLoaders.DataLoader(train_dataset, expt_config.train_batchsize),
        DataLoaders.DataLoader(test_dataset, expt_config.eval_batchsize),
    )
end

Base.length(d::CIFARDataContainer) = length(d.images)
Base.getindex(d::CIFARDataContainer, i::Int) = (Array(itemdata(apply(d.transform, d.images[i]))), d.labels[i])
MLDataPattern.getobs(d::CIFARDataContainer, i::Int64) = MLUtils.getobs(d, i)

# Validation
function validate(val_loader, model, ps, st, loss_function, args)
    batch_time = AverageMeter("Batch Time", "6.3f")
    data_time = AverageMeter("Data Time", "6.3f")
    losses = AverageMeter("Net Loss", "6.3f")
    loss1 = AverageMeter("Cross Entropy Loss", "6.3e")
    loss2 = AverageMeter("Skip Loss", "6.3e")
    residual = AverageMeter("Residual", "6.3e")
    top1 = AverageMeter("Accuracy", "3.2f")
    nfe = AverageMeter("NFE", "3.2f")

    progress = ProgressMeter(
        length(val_loader), (batch_time, data_time, losses, loss1, loss2, residual, top1, nfe), "Test:"
    )

    st_ = Lux.testmode(st)
    t = time()
    for (i, (x, y)) in enumerate(CUDA.functional() ? CuIterator(val_loader) : val_loader)
        B = size(x, ndims(x))
        data_time(time() - t, B)

        # Compute Output
        loss, st_, (ŷ, nfe_, celoss, skiploss, resi) = loss_function(x, y, model, ps, st_)
        st_ = Lux.update_state(st_, :update_mask, Val(true))

        # Measure Elapsed Time
        batch_time(time() - t, B)

        # Metrics
        acc1 = accuracy(cpu(ŷ), cpu(y))
        top1(acc1, B)
        nfe(nfe_, B)
        losses(loss, B)
        loss1(celoss, B)
        loss2(skiploss, B)
        residual(norm(resi), B)

        # Print Progress
        if i % args["print-freq"] == 0 || i == length(val_loader)
            should_log() && print_meter(progress, i)
        end

        t = time()
    end

    return (
        batch_time.sum,
        data_time.sum,
        loss1.sum,
        loss2.sum,
        losses.sum,
        nfe.sum,
        top1.sum,
        residual.sum,
        top1.count,
    )
end

# Training
function train_one_epoch(train_loader, model, ps, st, optimiser_state, epoch, loss_function, args)
    batch_time = AverageMeter("Batch Time", "6.3f")
    data_time = AverageMeter("Data Time", "6.3f")
    forward_pass_time = AverageMeter("Forward Pass Time", "6.3f")
    backward_pass_time = AverageMeter("Backward Pass Time", "6.3f")
    losses = AverageMeter("Net Loss", "6.3f")
    loss1 = AverageMeter("Cross Entropy Loss", "6.3e")
    loss2 = AverageMeter("Skip Loss", "6.3e")
    residual = AverageMeter("Residual", "6.3e")
    top1 = AverageMeter("Accuracy", "6.2f")
    nfe = AverageMeter("NFE", "6.2f")

    progress = ProgressMeter(
        length(train_loader),
        (batch_time, data_time, forward_pass_time, backward_pass_time, losses, loss1, loss2, residual, top1, nfe),
        "Epoch: [$epoch]",
    )

    st = Lux.trainmode(st)
    t = time()
    for (i, (x, y)) in enumerate(CuIterator(train_loader))
        B = size(x, ndims(x))
        data_time(time() - t, B)

        # Gradients and Update
        _t = time()
        (loss, st, (ŷ, nfe_, celoss, skiploss, resi)), back = Zygote.pullback(
            p -> loss_function(x, y, model, p, st), ps
        )
        forward_pass_time(time() - _t, B)
        _t = time()
        gs = back((one(loss), nothing, nothing))[1]
        backward_pass_time(time() - _t, B)
        st = Lux.update_state(st, :update_mask, Val(true))
        optimiser_state, ps = Optimisers.update(optimiser_state, ps, gs)

        # Measure Elapsed Time
        batch_time(time() - t, B)

        # Metrics
        acc1 = accuracy(cpu(ŷ), cpu(y))
        top1(acc1, B)
        nfe(nfe_, B)
        losses(loss, B)
        loss1(celoss, B)
        loss2(skiploss, B)
        residual(norm(resi), B)

        # Print Progress
        if i % args["print-freq"] == 0 || i == length(train_loader)
            should_log() && print_meter(progress, i)
        end

        t = time()
    end

    return (
        ps,
        st,
        optimiser_state,
        (
            batch_time.sum,
            data_time.sum,
            forward_pass_time.sum,
            backward_pass_time.sum,
            loss1.sum,
            loss2.sum,
            losses.sum,
            nfe.sum,
            top1.sum,
            residual.sum,
            top1.count,
        ),
    )
end

# Main Function
function get_base_experiment_name(args)
    return "data-CIFAR10_type-$(args["model-type"])_size-$(args["model-size"])_discrete-$(args["discrete"])_jfb-$(args["jfb"])"
end

function get_loggable_stats(stats)
    v = [stats...]
    is_distributed() && MPI.Reduce!(v, +, 0, MPI.COMM_WORLD)
    return v[1:end-1] ./ v[end]
end

function main(args)
    best_acc1 = 0

    # Seeding
    rng = Random.default_rng()
    Random.seed!(rng, args["seed"])

    # Model Construction
    expt_config = get_experiment_config(args)
    should_log() && println("$(now()) => creating model")
    model, ps, st = create_model(expt_config, args)

    should_log() && println("$(now()) => setting up dataloaders")
    train_loader, test_loader = get_dataloaders(expt_config)

    # Optimizer and Scheduler
    should_log() && println("$(now()) => creating optimiser")
    optimiser, scheduler = construct_optimiser(expt_config)
    optimiser_state = Optimisers.setup(optimiser, ps)
    if is_distributed()
        optimiser_state = FluxMPI.synchronize!(optimiser_state)
        should_log() && println("$(now()) ==> synced optimiser state across all ranks")
    end

    if args["resume"] != ""
        if isfile(args["resume"])
            checkpoint = deserialize(args["resume"])
            args["start-epoch"] = checkpoint["epoch"]
            optimiser_state = gpu(checkpoint["optimiser_state"])
            ps = gpu(checkpoint["model_parameters"])
            st = gpu(checkpoint["model_states"])
            should_log() && println("$(now()) => loaded checkpoint `$(args["resume"])` (epoch $(args["start-epoch"]))")
        else
            should_log() && println("$(now()) => no checkpoint found at `$(args["resume"])`. Starting from scratch.")
        end
    end

    loss_function = get_loss_function(args)

    if args["evaluate"]
        validate(test_loader, model, ps, st, loss_function, args)
        return nothing
    end

    invoke_gc()

    expt_name = get_base_experiment_name(args)
    store_in = args["expt-subdir"] == "" ? string(now()) : args["expt-subdir"]

    ckpt_dir = joinpath(args["checkpoint-dir"], expt_name, store_in)
    log_path = joinpath(args["log-dir"], expt_name, store_in, "results.csv")

    should_log() && println("$(now()) => checkpoint directory `$(ckpt_dir)`")

    csv_logger = CSVLogger(log_path, ["Epoch", "Train/Batch Time", "Train/Data Time", "Train/Forward Pass Time", "Train/Backward Pass Time", "Train/Cross Entropy Loss", "Train/Skip Loss", "Train/Net Loss", "Train/NFE", "Train/Accuracy", "Train/Residual", "Test/Batch Time", "Test/Data Time", "Test/Cross Entropy Loss", "Test/Skip Loss", "Test/Net Loss", "Test/NFE", "Test/Accuracy", "Test/Residual"])

    should_log() && println("$(now()) => logging results to `$(log_path)`")

    should_log() && serialize(joinpath(dirname(log_path), "setup.jls"), Dict("config" => expt_config, "args" => args))

    st = hasproperty(expt_config, :pretrain_epochs) && getproperty(expt_config, :pretrain_epochs) > 0 ? Lux.update_state(st, :fixed_depth, Val(getproperty(expt_config, :num_layers))) : st

    for epoch in args["start-epoch"]:(expt_config.nepochs)
        # Train for 1 epoch
        ps, st, optimiser_state, train_stats = train_one_epoch(
            train_loader, model, ps, st, optimiser_state, epoch, loss_function, args
        )
        train_stats = get_loggable_stats(train_stats)

        should_log() && println()

        # Some Housekeeping
        invoke_gc()

        # Evaluate on validation set
        val_stats = validate(test_loader, model, ps, st, loss_function, args)
        val_stats = get_loggable_stats(val_stats)

        should_log() && println()

        csv_logger(epoch, train_stats..., val_stats...)
        should_log() && println("$(now()) => logged intermediated results to csv file\n")

        # ParameterSchedulers
        eta_new = ParameterSchedulers.next!(scheduler)
        optimiser_state = update_lr(optimiser_state, eta_new)
        if hasproperty(expt_config, :pretrain_epochs) && getproperty(expt_config, :pretrain_epochs) == epoch
            should_log() && println("$(now()) => pretraining completed\n")
            st = Lux.update_state(st, :fixed_depth, Val(0))
        end

        # Some Housekeeping
        invoke_gc()

        # Remember Best Accuracy and Save Checkpoint
        is_best = val_stats[1] > best_acc1
        best_acc1 = max(val_stats[1], best_acc1)

        save_state = Dict(
            "epoch" => epoch,
            "config" => expt_config,
            "accuracy" => accuracy,
            "model_states" => cpu(st),
            "model_parameters" => cpu(ps),
            "optimiser_state" => cpu(optimiser_state),
        )
        save_checkpoint(save_state, is_best, joinpath(ckpt_dir, "checkpoint.jls"))
    end
end

main(parse_commandline_arguments())
