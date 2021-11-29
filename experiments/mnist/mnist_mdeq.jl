# Load Packages
using CUDA,
    Dates,
    DiffEqSensitivity,
    FastDEQ,
    Flux,
    MLDatasets,
    OrdinaryDiffEq,
    Statistics,
    SteadyStateDiffEq,
    Plots,
    Random,
    Wandb,
    Zygote

## Models
function get_model(
    maxiters::Int,
    abstol::T,
    reltol::T,
    batch_size::Int,
    model_type::String,
) where {T}
    main_layers =
        (
            BasicResidualBlock((28, 28), 8, 8),
            BasicResidualBlock((14, 14), 16, 16),
            BasicResidualBlock((7, 7), 32, 32),
        ) .|> gpu
    mapping_layers =
        [
            identity downsample_module(8, 16, 28, 14) downsample_module(8, 32, 28, 7)
            upsample_module(16, 8, 14, 28) identity downsample_module(16, 32, 14, 7)
            upsample_module(32, 8, 7, 28) upsample_module(32, 16, 7, 14) identity
        ] .|> gpu
    model = DEQChain(
        expand_channels_module(1, 8),
        (
            model_type == "vanilla" ? MultiScaleDeepEquilibriumNetwork :
            MultiScaleSkipDeepEquilibriumNetwork
        )(
            main_layers,
            mapping_layers,
            get_default_dynamicss_solver(abstol, reltol),
            # get_default_ssrootfind_solver(abstol, reltol, LimitedMemoryBroydenSolver;
            #                               device = gpu, original_dims = (1, (28 * 28 * 8) + (14 * 14 * 16) + (7 * 7 * 32)),
            #                               batch_size = batch_size, maxiters = maxiters),
            maxiters = maxiters,
            sensealg = get_default_ssadjoint(abstol, reltol, maxiters),
            verbose = false,
        ),
        t -> tuple(t...),
        Parallel(
            +,
            downsample_module(8, 32, 28, 7),
            downsample_module(16, 32, 14, 7),
            identity,
        ),
        Flux.flatten,
        Dense(7 * 7 * 32, 10; bias = true),
    )
    return model |> gpu
end


## Utilities
function register_nfe_counts(model, buffer)
    callback() = push!(buffer, get_and_clear_nfe!(model))
    return callback
end

function loss_and_accuracy(model, dataloader)
    matches, total_loss, total_datasize, total_nfe = 0, 0, 0, 0
    for (x, y) in dataloader
        x = x |> gpu
        y = y |> gpu

        ŷ = model(x)
        ŷ = ŷ isa Tuple ? ŷ[1] : ŷ  # Handle SkipDEQ
        total_nfe += get_and_clear_nfe!(model) * size(x, ndims(x))
        total_loss += Flux.Losses.logitcrossentropy(ŷ, y) * size(x, ndims(x))
        matches += sum(argmax.(eachcol(ŷ)) .== Flux.onecold(y |> cpu))
        total_datasize += size(x, ndims(x))
    end
    return (
        total_loss / total_datasize,
        matches / total_datasize,
        total_nfe / total_datasize,
    )
end


## Training Function
function train(config::Dict)
    ## Setup Logging & Experiment Configuration
    lg = WandbLogger(
        project = "FastDEQ.jl",
        name = "fastdeqjl-supervised_mnist_classication-mdeq-$(now())",
        config = config,
    )

    ## Reproducibility
    Random.seed!(get_config(lg, "seed"))

    ## Dataset
    batch_size = get_config(lg, "batch_size")
    eval_batch_size = get_config(lg, "eval_batch_size")

    _xs_train, _ys_train = MNIST.traindata(Float32)
    _xs_test, _ys_test = MNIST.testdata(Float32)

    xs_train, ys_train = Flux.unsqueeze(_xs_train, 3), Float32.(Flux.onehotbatch(_ys_train, 0:9))
    xs_test, ys_test = Flux.unsqueeze(_xs_test, 3), Float32.(Flux.onehotbatch(_ys_test, 0:9))

    traindata = (xs_train, ys_train)
    trainiter = Flux.Data.DataLoader(
        traindata;
        batchsize = batch_size,
        shuffle = true,
    )
    testiter = Flux.Data.DataLoader(
        (xs_test, ys_test);
        batchsize = eval_batch_size,
        shuffle = false,
    )

    ## Model Setup
    model = get_model(
        get_config(lg, "maxiters"),
        Float32(get_config(lg, "abstol")),
        Float32(get_config(lg, "reltol")),
        batch_size,
        get_config(lg, "model_type"),
    )

    loss_function =
        SupervisedLossContainer(Flux.Losses.logitcrossentropy, 1.0f0)

    ## Warmup
    __x = rand(28, 28, 1, 1) |> gpu
    __y = Flux.onehotbatch([1], 0:9) |> gpu
    Flux.gradient(() -> loss_function(model, __x, __y), Flux.params(model))

    nfe_counts = []
    cb = register_nfe_counts(model, nfe_counts)

    ## Training Loop
    ps = Flux.params(model)
    opt = ADAM(get_config(lg, "learning_rate"))
    step = 1
    for epoch = 1:get_config(lg, "epochs")
        try
            for (x, y) in trainiter
                x = x |> gpu
                y = y |> gpu

                loss, back =
                    Zygote.pullback(() -> loss_function(model, x, y), ps)
                gs = back(one(loss))
                Flux.Optimise.update!(opt, ps, gs)

                ### Store the NFE Count
                cb()

                ### Log the losses
                log(
                    lg,
                    Dict(
                        "Training/Step/Loss" => loss,
                        "Training/Step/NFE" => nfe_counts[end],
                        "Training/Step/Count" => step,
                    ),
                )
                step += 1
            end

            ### Training Loss/Accuracy
            train_loss, train_acc, train_nfe = loss_and_accuracy(
                model,
                Flux.Data.DataLoader(traindata; batchsize = eval_batch_size, shuffle = false),
            )
            log(
                lg,
                Dict(
                    "Training/Epoch/Count" => epoch,
                    "Training/Epoch/Loss" => train_loss,
                    "Training/Epoch/NFE" => train_nfe,
                    "Training/Epoch/Accuracy" => train_acc,
                ),
            )

            ### Testing Loss/Accuracy
            test_loss, test_acc, test_nfe = loss_and_accuracy(model, testiter)
            log(
                lg,
                Dict(
                    "Testing/Epoch/Count" => epoch,
                    "Testing/Epoch/Loss" => test_loss,
                    "Testing/Epoch/NFE" => test_nfe,
                    "Testing/Epoch/Accuracy" => test_acc,
                ),
            )
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

    close(lg)

    return model, nfe_counts
end

## Plotting
function plot_nfe_counts(nfe_counts_1, nfe_counts_2)
    p = plot(nfe_counts_1, label = "Vanilla DEQ")
    plot!(p, nfe_counts_2, label = "Skip DEQ")
    xlabel!(p, "Training Iteration")
    ylabel!(p, "NFE Count")
    title!(p, "NFE over Training Iterations of DEQ vs SkipDEQ")
    return p
end

## Run Experiment
nfe_count_dict = Dict("vanilla" => [], "skip" => [])

for seed in [1, 11, 111]
    for model_type in ["vanilla", "skip"]
        config = Dict(
            "seed" => seed,
            "learning_rate" => 0.001,
            "abstol" => 0.1f0,
            "reltol" => 0.1f0,
            "maxiters" => 10,
            "epochs" => 25,
            "batch_size" => 512,
            "eval_batch_size" => 512,
            "model_type" => model_type,
        )

        model, nfe_counts = train(config)

        push!(nfe_count_dict[model_type], nfe_counts)
    end
end

plot_nfe_counts(
    vec(mean(hcat(nfe_count_dict["vanilla"]...), dims = 2)),
    vec(mean(hcat(nfe_count_dict["skip"]...), dims = 2)),
)
