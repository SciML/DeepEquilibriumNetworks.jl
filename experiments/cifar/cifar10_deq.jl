# Load Packages
using CUDA, Dates, DiffEqSensitivity, FastDEQ, Flux, FluxMPI, OrdinaryDiffEq, Serialization, Statistics,
      SteadyStateDiffEq, MLDatasets, MPI, Plots, Random, ParameterSchedulers, Wandb, Zygote
using ParameterSchedulers: Scheduler, Cos
using MLDataPattern: splitobs, shuffleobs

MPI.Init()
CUDA.allowscalar(false)

const MPI_COMM_WORLD = MPI.COMM_WORLD
const MPI_COMM_SIZE = MPI.Comm_size(MPI_COMM_WORLD)

## Models
function get_model(maxiters::Int, abstol::T, reltol::T, dropout_rate::Real, model_type::String, batch_size::Int,
                   solver_type::String="dynamicss") where {T}
    main_layers = (BasicResidualBlock((32, 32), 8, 8; dropout_rate=dropout_rate),
                   BasicResidualBlock((16, 16), 16, 16; dropout_rate=dropout_rate),
                   BasicResidualBlock((8, 8), 32, 32; dropout_rate=dropout_rate))
    mapping_layers = [identity downsample_module(8, 16, 32, 16) downsample_module(8, 32, 32, 8)
                      upsample_module(16, 8, 16, 32) identity downsample_module(16, 32, 16, 8)
                      upsample_module(32, 8, 8, 32) upsample_module(32, 16, 8, 16) identity]
    solver = solver_type == "dynamicss" ? get_default_dynamicss_solver(abstol, reltol) :
             get_default_ssrootfind_solver(abstol, reltol, LimitedMemoryBroydenSolver; device=gpu,
                                           original_dims=(1, (28 * 28 * 8) + (14 * 14 * 16) + (7 * 7 * 32)),
                                           batch_size=batch_size, maxiters=maxiters)

    if model_type == "skip"
        deq = MultiScaleSkipDeepEquilibriumNetwork(main_layers, mapping_layers,
                                                   (BasicResidualBlock((32, 32), 8, 8),
                                                    downsample_module(8, 16, 32, 16), downsample_module(8, 32, 32, 8)),
                                                   solver; maxiters=maxiters,
                                                   sensealg=get_default_ssadjoint(abstol, reltol, maxiters),
                                                   verbose=false)
    else
        _deq = model_type == "vanilla" ? MultiScaleDeepEquilibriumNetwork : MultiScaleSkipDeepEquilibriumNetwork
        deq = _deq(main_layers, mapping_layers, solver; maxiters=maxiters,
                   sensealg=get_default_ssadjoint(abstol, reltol, maxiters), verbose=false)
    end
    model = DEQChain(expand_channels_module(3, 8), deq, t -> tuple(t...),
                     Parallel(+, downsample_module(8, 32, 32, 8), downsample_module(16, 32, 16, 8),
                              expand_channels_module(32, 32)), Flux.flatten, Dense(8 * 8 * 32, 10; bias=true))
    if MPI_COMM_SIZE > 1
        return DataParallelFluxModel(model, [i % length(CUDA.devices()) for i in 1:MPI_COMM_SIZE])
    else
        return gpu(model)
    end
end

## Utilities
function register_nfe_counts(model, buffer)
    callback() = push!(buffer, get_and_clear_nfe!(model))
    return callback
end

function loss_and_accuracy(model, dataloader)
    matches, total_loss, total_datasize, total_nfe = 0, 0, 0, 0
    for (x, y) in dataloader
        x = gpu(x)
        y = gpu(y)

        ŷ = model(x)
        ŷ = ŷ isa Tuple ? ŷ[1] : ŷ  # Handle SkipDEQ
        total_nfe += get_and_clear_nfe!(model) * size(x, ndims(x))
        total_loss += Flux.Losses.logitcrossentropy(ŷ, y) * size(x, ndims(x))
        matches += sum(argmax.(eachcol(ŷ)) .== Flux.onecold(cpu(y)))
        total_datasize += size(x, ndims(x))
    end
    return (total_loss / total_datasize, matches / total_datasize, total_nfe / total_datasize)
end

## Training Function
function train(config::Dict)
    comm = MPI_COMM_WORLD
    rank = MPI.Comm_rank(comm)

    ## Setup Logging & Experiment Configuration
    expt_name = "fastdeqjl-supervised_cifar10_classication-$(now())"
    lg_wandb = WandbLoggerMPI(; project="FastDEQ.jl", name=expt_name, config=config)
    lg_term = PrettyTableLogger("logs/" * expt_name * ".log",
                                ["Epoch Number", "Train/NFE", "Train/Accuracy", "Train/Loss", "Train/Time", "Test/NFE",
                                 "Test/Accuracy", "Test/Loss", "Test/Time"],
                                ["Train/Running/NFE", "Train/Running/Loss"])

    ## Reproducibility
    Random.seed!(get_config(lg_wandb, "seed"))

    ## Model Setup
    model = get_model(get_config(lg_wandb, "maxiters"), Float32(get_config(lg_wandb, "abstol")),
                      Float32(get_config(lg_wandb, "reltol")), Float64(get_config(lg_wandb, "dropout_rate")),
                      get_config(lg_wandb, "model_type"), get_config(lg_wandb, "batch_size"),
                      get_config(lg_wandb, "solver_type"))

    ## Dataset
    batch_size = get_config(lg_wandb, "batch_size")
    eval_batch_size = get_config(lg_wandb, "eval_batch_size")

    _xs_train, _ys_train = CIFAR10.traindata(Float32)
    _xs_test, _ys_test = CIFAR10.testdata(Float32)

    xs_train, ys_train = _xs_train, Float32.(Flux.onehotbatch(_ys_train, 0:9))
    xs_test, ys_test = _xs_test, Float32.(Flux.onehotbatch(_ys_test, 0:9))

    traindata = (xs_train, ys_train)
    trainiter = DataParallelDataLoader(traindata; batchsize=batch_size, shuffle=true)
    testiter = DataParallelDataLoader((xs_test, ys_test); batchsize=eval_batch_size, shuffle=false)

    ## Loss Function
    loss_function = SupervisedLossContainer(Flux.Losses.logitcrossentropy, 2.5f0)

    nfe_counts = []
    cb = register_nfe_counts(model, nfe_counts)

    ## Warmup with a smaller batch
    _x_warmup, _y_warmup = gpu(rand(32, 32, 3, 1)), gpu(Flux.onehotbatch([1], 0:9))
    loss_function(model, _x_warmup, _y_warmup)
    @info "Rank $rank: Forward Pass Warmup Completed"
    Zygote.gradient(() -> loss_function(model, _x_warmup, _y_warmup), Flux.params(model))
    @info "Rank $rank: Warmup Completed"

    ## Training Loop
    ps = Flux.params(model)
    opt = Scheduler(Cos(get_config(lg_wandb, "learning_rate"), 1e-5,
                        length(trainiter) * get_config(lg_wandb, "epochs")),
                    ADAM(get_config(lg_wandb, "learning_rate"), (0.9, 0.999)))
    step = 1
    train_vec = zeros(3)
    test_vec = zeros(3)

    datacount_trainiter = length(trainiter.indices)
    datacount_testiter = length(testiter.indices)
    datacount_trainiter_total = size(xs_train, ndims(xs_train))
    datacount_testiter_total = size(xs_test, ndims(xs_test))

    @info "Rank $rank: [ $datacount_trainiter / $datacount_trainiter_total ] Training Images | [ $datacount_testiter / $datacount_testiter_total ] Test Images"

    for epoch in 1:get_config(lg_wandb, "epochs")
        try
            epoch_start_time = time()
            for (x, y) in trainiter
                x = gpu(x)
                y = gpu(y)

                _res = Zygote.withgradient(() -> loss_function(model, x, y), ps)
                loss = _res.val
                gs = _res.grad

                Flux.Optimise.update!(opt, ps, gs)

                ### Store the NFE Count
                cb()

                ### Log the losses
                log(lg_wandb,
                    Dict("Training/Step/Loss" => loss, "Training/Step/NFE" => nfe_counts[end],
                         "Training/Step/Count" => step))
                lg_term(; records=Dict("Train/Running/NFE" => nfe_counts[end], "Train/Running/Loss" => loss))
                step += 1
            end
            epoch_end_time = time()

            ### Training Loss/Accuracy
            train_loss, train_acc, train_nfe = loss_and_accuracy(model, trainiter)

            if MPI_COMM_SIZE > 1
                train_vec .= [train_loss, train_acc, train_nfe] .* datacount_trainiter
                safe_reduce!(train_vec, +, 0, comm)
                train_loss, train_acc, train_nfe = train_vec ./ datacount_trainiter_total
            end

            log(lg_wandb,
                Dict("Training/Epoch/Count" => epoch, "Training/Epoch/Loss" => train_loss,
                     "Training/Epoch/NFE" => train_nfe, "Training/Epoch/Accuracy" => train_acc))

            ### Testing Loss/Accuracy
            test_start_time = time()
            test_loss, test_acc, test_nfe = loss_and_accuracy(model, testiter)
            test_end_time = time()

            if MPI_COMM_SIZE > 1
                test_vec .= [test_loss, test_acc, test_nfe] .* datacount_trainiter
                safe_reduce!(test_vec, +, 0, comm)
                test_loss, test_acc, test_nfe = test_vec ./ datacount_trainiter_total
            end

            log(lg_wandb,
                Dict("Testing/Epoch/Count" => epoch, "Testing/Epoch/Loss" => test_loss, "Testing/Epoch/NFE" => test_nfe,
                     "Testing/Epoch/Accuracy" => test_acc))
            lg_term(epoch, train_nfe, train_acc, train_loss, epoch_end_time - epoch_start_time, test_nfe, test_acc,
                    test_loss, test_end_time - test_start_time)

            MPI.Barrier(comm)
        catch ex
            if ex isa Flux.Optimise.StopException
                break
            elseif ex isa Flux.Optimise.SkipException
                continue
            else
                rethrow(ex)
            end
        end
    end

    close(lg_wandb)
    close(lg_term)

    return model, nfe_counts
end

## Run Experiment
nfe_count_dict = Dict("vanilla" => [], "skip" => [], "skip_no_extra_params" => [])

for seed in [1, 11, 111]
    for model_type in ["skip_no_extra_params", "vanilla", "skip"]
        config = Dict("seed" => seed, "learning_rate" => 0.001, "abstol" => 1f-1, "reltol" => 1f-1, "maxiters" => 20,
                      "epochs" => 50, "dropout_rate" => 0.10, "batch_size" => 256, "eval_batch_size" => 256,
                      "model_type" => model_type, "solver_type" => "dynamicss")

        model, nfe_counts = train(config)

        push!(nfe_count_dict[model_type], nfe_counts)
    end
end

if MPI.Comm_rank(MPI_COMM_WORLD) == 0
    filename = "fastdeqjl-supervised_cifar10_classication-$(now()).jls"
    serialize(joinpath("artifacts", filename), nfe_count_dict)
    @info "Serialized NFE Counts to $filename"
end
