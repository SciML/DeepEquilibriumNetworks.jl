## TODO: Update for MultiScale DEQ

# Load Packages
using CUDA, Dates, DiffEqSensitivity, FastDEQ, Flux, FluxMPI, OrdinaryDiffEq, Statistics, SteadyStateDiffEq, MLDatasets,
      MPI, Plots, Random, ParameterSchedulers, Wandb, Zygote
using ParameterSchedulers: Scheduler, Cos
using MLDataPattern: splitobs, shuffleobs

MPI.Init()
CUDA.allowscalar(false)

const MPI_COMM_WORLD = MPI.COMM_WORLD
const MPI_COMM_SIZE = MPI.Comm_size(MPI_COMM_WORLD)

## Models
function get_model(maxiters::Int, abstol::T, reltol::T, dropout_rate::Real, model_type::String) where {T}
    model = WidthStackedDEQ([expand_channels_module(3, 8), downsample_module(8, 16, 32, 16),
                             downsample_module(16, 32, 16, 8)],
                            [model_type == "skip" ?
                             SkipDeepEquilibriumNetwork(BasicResidualBlock((32 ÷ (2^(i - 1)), 32 ÷ (2^(i - 1))),
                                                                           2^(i + 2), 2^(i + 2);
                                                                           dropout_rate=dropout_rate),
                                                        BasicResidualBlock((32 ÷ (2^(i - 1)), 32 ÷ (2^(i - 1))),
                                                                           2^(i + 2), 2^(i + 2);
                                                                           dropout_rate=dropout_rate),
                                                        get_default_dynamicss_solver(reltol, abstol, Tsit5());
                                                        maxiters=maxiters,
                                                        sensealg=get_default_ssadjoint(reltol, abstol, maxiters),
                                                        verbose=false) :
                             DeepEquilibriumNetwork(BasicResidualBlock((32 ÷ (2^(i - 1)), 32 ÷ (2^(i - 1))), 2^(i + 2),
                                                                       2^(i + 2); dropout_rate=dropout_rate),
                                                    get_default_dynamicss_solver(reltol, abstol, Tsit5());
                                                    maxiters=maxiters,
                                                    sensealg=get_default_ssadjoint(reltol, abstol, maxiters),
                                                    verbose=false) for i in 1:3],
                            [downsample_module(8, 32, 32, 8), downsample_module(16, 32, 16, 8),
                             expand_channels_module(32, 32)], (x...) -> foldl(+, x),
                            Chain(Flux.flatten, Dense(8 * 8 * 32, 10)))
    if MPI_COMM_SIZE > 1
        return DataParallelFluxModel(model, [i % length(CUDA.devices()) for i in 1:MPI.Comm_size(MPI.COMM_WORLD)])
    else
        return gpu(model)
    end
end

## Utilities
function register_nfe_counts(model, buffer)
    callback() = push!(buffer, [get_and_clear_nfe!(model)...])
    return callback
end

function loss_and_accuracy(model, dataloader)
    matches, total_loss, total_datasize, total_nfe = 0, 0, 0, [0, 0, 0]
    for (x, y) in dataloader
        x = gpu(x)
        y = gpu(y)

        ŷ = model(x)
        ŷ = ŷ isa Tuple ? ŷ[1] : ŷ  # Handle SkipDEQ
        total_nfe .+= get_and_clear_nfe!(model) .* size(x, ndims(x))
        total_loss += Flux.Losses.logitcrossentropy(ŷ, y) * size(x, ndims(x))
        matches += sum(argmax.(eachcol(ŷ)) .== Flux.onecold(cpu(y)))
        total_datasize += size(x, ndims(x))
    end
    return (total_loss / total_datasize, matches / total_datasize, total_nfe ./ total_datasize)
end

## Training Function
function train(config::Dict)
    comm = MPI_COMM_WORLD
    rank = MPI.Comm_rank(comm)

    ## Setup Logging & Experiment Configuration
    lg_wandb = WandbLoggerMPI(; project="FastDEQ.jl", name="fastdeqjl-supervised_cifar10_classication-$(now())",
                              config=config)
    lg_term = PrettyTableLogger("logs/fastdeqjl-supervised_cifar10_classication-$(now()).log",
                                ["Epoch Number", "Train/NFE1", "Train/NFE2", "Train/NFE3", "Train/Accuracy",
                                 "Train/Loss", "Test/NFE1", "Test/NFE2", "Test/NFE3", "Test/Accuracy", "Test/Loss"],
                                ["Train/Running/NFE1", "Train/Running/NFE2", "Train/Running/NFE3",
                                 "Train/Running/Loss"])

    ## Reproducibility
    Random.seed!(get_config(lg_wandb, "seed"))

    ## Model Setup
    model = get_model(get_config(lg_wandb, "maxiters"), Float32(get_config(lg_wandb, "abstol")),
                      Float32(get_config(lg_wandb, "reltol")), Float64(get_config(lg_wandb, "dropout_rate")),
                      get_config(lg_wandb, "model_type"))

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
    loss_function = SupervisedLossContainer(Flux.Losses.logitcrossentropy, 0.1f0)

    nfe_counts = Vector{Int64}[]
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
    train_vec = zeros(Float32, 5)
    test_vec = zeros(Float32, 5)

    datacount_trainiter = length(trainiter.indices)
    datacount_testiter = length(testiter.indices)
    datacount_trainiter_total = size(xs_train, ndims(xs_train))
    datacount_testiter_total = size(xs_test, ndims(xs_test))

    @info "Rank $rank: [ $datacount_trainiter / $datacount_trainiter_total ] Training Images | [ $datacount_testiter / $datacount_testiter_total ] Test Images"

    for epoch in 1:get_config(lg_wandb, "epochs")
        try
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
                    Dict("Training/Step/Loss" => loss, "Training/Step/NFE1" => nfe_counts[end][1],
                         "Training/Step/NFE2" => nfe_counts[end][2], "Training/Step/NFE3" => nfe_counts[end][3],
                         "Training/Step/Count" => step))
                lg_term(;
                        records=Dict("Train/Running/NFE1" => nfe_counts[end][1],
                                     "Train/Running/NFE2" => nfe_counts[end][2],
                                     "Train/Running/NFE3" => nfe_counts[end][3], "Train/Running/Loss" => loss))
                step += 1
            end

            ### Training Loss/Accuracy
            train_loss, train_acc, train_nfe = loss_and_accuracy(model, trainiter)

            if MPI_COMM_SIZE > 1
                train_vec[1] = train_loss * datacount_trainiter
                train_vec[2] = train_acc * datacount_trainiter
                train_vec[3:end] .= train_nfe .* datacount_trainiter
                safe_reduce!(train_vec, +, 0, comm)
                train_loss, train_acc, train_nfe = (train_vec[1] / datacount_trainiter_total,
                                                    train_vec[2] / datacount_trainiter_total,
                                                    train_vec[3:end] ./ datacount_trainiter_total)
            end

            log(lg_wandb,
                Dict("Training/Epoch/Count" => epoch, "Training/Epoch/Loss" => train_loss,
                     "Training/Epoch/NFE1" => train_nfe[1], "Training/Epoch/NFE2" => train_nfe[2],
                     "Training/Epoch/NFE3" => train_nfe[3], "Training/Epoch/Accuracy" => train_acc))

            ### Testing Loss/Accuracy
            test_loss, test_acc, test_nfe = loss_and_accuracy(model, testiter)

            if MPI_COMM_SIZE > 1
                test_vec[1] = test_loss * datacount_testiter
                test_vec[2] = test_acc * datacount_testiter
                test_vec[3:end] .= test_nfe .* datacount_testiter
                safe_reduce!(test_vec, +, 0, comm)
                test_loss, test_acc, test_nfe = (test_vec[1] / datacount_testiter_total,
                                                 test_vec[2] / datacount_testiter_total,
                                                 test_vec[3:end] ./ datacount_testiter_total)
            end

            log(lg_wandb,
                Dict("Testing/Epoch/Count" => epoch, "Testing/Epoch/Loss" => test_loss,
                     "Testing/Epoch/NFE1" => test_nfe[1], "Testing/Epoch/NFE2" => test_nfe[2],
                     "Testing/Epoch/NFE3" => test_nfe[3], "Testing/Epoch/Accuracy" => test_acc))

            lg_term(epoch, train_nfe..., train_acc, train_loss, test_nfe..., test_acc, test_loss)

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

## Plotting
function plot_nfe_counts(nfe_counts_1, nfe_counts_2)
    p = plot(nfe_counts_1; label="Vanilla DEQ")
    plot!(p, nfe_counts_2; label="Skip DEQ")
    xlabel!(p, "Training Iteration")
    ylabel!(p, "NFE Count")
    title!(p, "NFE over Training Iterations of DEQ vs SkipDEQ")
    return p
end

## Run Experiment
nfe_count_dict = Dict("vanilla" => [], "skip" => [])

for seed in [1, 11, 111]
    for model_type in ["skip", "vanilla"]
        config = Dict("seed" => seed, "learning_rate" => 0.001, "abstol" => 1f-1, "reltol" => 1f-1, "maxiters" => 20,
                      "epochs" => 50, "dropout_rate" => 0.10, "batch_size" => 128, "eval_batch_size" => 128,
                      "model_type" => model_type)

        model, nfe_counts = train(config)

        push!(nfe_count_dict[model_type], nfe_counts)
    end
end

plot_nfe_counts(vec(mean(hcat(nfe_count_dict["vanilla"]...); dims=2)),
                vec(mean(hcat(nfe_count_dict["skip"]...); dims=2)))
