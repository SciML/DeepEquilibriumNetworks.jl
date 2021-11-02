# Load Packages
using CUDA,
    Dates,
    DiffEqSensitivity,
    FastDEQ,
    Flux,
    OrdinaryDiffEq,
    Statistics,
    SteadyStateDiffEq,
    Plots,
    Random,
    Wandb,
    Zygote
using DataLoaders: DataLoader
using MLDataPattern: splitobs, shuffleobs

### Need to allow for gradient accumulation
CUDA.allowscalar(true)

## Models
# Resnet Layer
struct ResNetLayer{C1,C2,N1,N2,N3}
    conv1::C1
    conv2::C2
    norm1::N1
    norm2::N2
    norm3::N3
end

Flux.@functor ResNetLayer

function ResNetLayer(
    n_channels::Int,
    n_inner_channels::Int;
    kernel_size::Tuple{Int,Int} = (3, 3),
    num_groups::Int = 8,
    affine::Bool = false,
)
    conv1 = Conv(
        kernel_size,
        n_channels => n_inner_channels,
        relu;
        pad = kernel_size .÷ 2,
        bias = false,
        init = (dims...) -> randn(Float32, dims...) .* 0.01f0,
    )
    conv2 = Conv(
        kernel_size,
        n_inner_channels => n_channels;
        pad = kernel_size .÷ 2,
        bias = false,
        init = (dims...) -> randn(Float32, dims...) .* 0.01f0,
    )
    norm1 = GroupNorm(n_inner_channels, num_groups, affine = affine)
    norm2 = GroupNorm(n_channels, num_groups, affine = affine)
    norm3 = GroupNorm(n_channels, num_groups, affine = affine)

    return ResNetLayer(conv1, conv2, norm1, norm2, norm3)
end

(rl::ResNetLayer)(z, x) =
    rl.norm3(relu.(z .+ rl.norm2(x .+ rl.conv2(rl.norm1(rl.conv1(z))))))

(rl::ResNetLayer)(x) =
    rl.norm3(relu.(rl.norm2(rl.conv2(rl.norm1(rl.conv1(x))))))


struct MultiScaleCombinationLayer{Op,L1,L2,L3,L4}
    op::Op
    layer1::L1
    layer2::L2
    layer3::L3
    layer4::L4
end

Flux.@functor MultiScaleCombinationLayer

function (mscl::MultiScaleCombinationLayer)(x)
    x1 = mscl.layer1(x[1])
    x2 = mscl.layer2(x[2])
    x3 = mscl.layer3(x[3])
    x4 = mscl.layer4(x[4])
    return mscl.op(x1, x2, x3, x4)
end


function get_model(
    maxiters::Int,
    abstol::T,
    reltol::T,
    model_type::String,
) where {T}
    main_layers = (
        ResNetLayer(8, 16) |> gpu,
        ResNetLayer(8, 16) |> gpu,
        ResNetLayer(8, 16) |> gpu,
        ResNetLayer(8, 16) |> gpu,
    )
    mapping_layers = (
        (identity, MeanPool((2, 2)), MeanPool((4, 4)), MeanPool((8, 8))) .|>
        gpu,
        (
            ConvTranspose((4, 4), 8 => 8, stride = (2, 2), pad = SamePad()),
            identity,
            MeanPool((2, 2)),
            MeanPool((4, 4)),
        ) .|> gpu,
        (
            Chain(
                ConvTranspose((4, 4), 8 => 8, stride = (2, 2), pad = SamePad()),
                ConvTranspose((4, 4), 8 => 8, stride = (2, 2), pad = SamePad()),
            ),
            ConvTranspose((4, 4), 8 => 8, stride = (2, 2), pad = SamePad()),
            identity,
            MeanPool((2, 2)),
        ) .|> gpu,
        (
            Chain(
                ConvTranspose((3, 3), 8 => 8, stride = (2, 2), pad = 0),
                ConvTranspose((4, 4), 8 => 8, stride = (2, 2), pad = SamePad()),
                ConvTranspose((4, 4), 8 => 8, stride = (2, 2), pad = SamePad()),
            ),
            Chain(
                ConvTranspose((3, 3), 8 => 8, stride = (2, 2), pad = 0),
                ConvTranspose((4, 4), 8 => 8, stride = (2, 2), pad = SamePad()),
            ),
            ConvTranspose((3, 3), 8 => 8, stride = (2, 2), pad = 0),
            identity,
        ) .|> gpu,
    )
    model = DEQChain(
        Chain(
            Conv((3, 3), 1 => 8, relu; bias = true, pad = 1),
            BatchNorm(8, affine = true),
        ) |> gpu,
        model_type == "vanilla" ?
            MultiScaleDeepEquilibriumNetworkS4(
                main_layers,
                mapping_layers,
                DynamicSS(Tsit5(); abstol = abstol, reltol = reltol),
                maxiters = maxiters,
                sensealg = SteadyStateAdjoint(
                    autodiff = true,
                    autojacvec = ZygoteVJP(),
                    linsolve = LinSolveKrylovJL(
                        rtol = reltol,
                        atol = abstol,
                        itmax = maxiters,
                    ),
                ),
                verbose = false,
            ) :
            MultiScaleSkipDeepEquilibriumNetworkS4(
                main_layers,
                mapping_layers,
                (
                    ResNetLayer(8, 16) |> gpu,
                    Chain(MeanPool((2, 2)), ResNetLayer(8, 16)) |> gpu,
                    Chain(MeanPool((4, 4)), ResNetLayer(8, 16)) |> gpu,
                    Chain(MeanPool((8, 8)), ResNetLayer(8, 16)) |> gpu,
                ),
                DynamicSS(Tsit5(); abstol = abstol, reltol = reltol),
                maxiters = maxiters,
                sensealg = SteadyStateAdjoint(
                    autodiff = true,
                    autojacvec = ZygoteVJP(),
                    linsolve = LinSolveKrylovJL(
                        rtol = reltol,
                        atol = abstol,
                        itmax = maxiters,
                    ),
                ),
                verbose = false,
            ),
        MultiScaleCombinationLayer(
            (args...) -> foldl(+, args),
            MeanPool((8, 8)),
            MeanPool((4, 4)),
            MeanPool((2, 2)),
            identity,
        ) |> gpu,
        Flux.flatten,
        Dense(3 * 3 * 8, 10; bias = true) |> gpu,
    )
    return model
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
    xs_train, ys_train = (
        # convert each image into h*w*1 array of floats 
        [
            Float32.(reshape(img, 28, 28, 1)) for
            img in Flux.Data.MNIST.images(:train)
        ],
        # one-hot encode the labels
        [Float32.(Flux.onehot(y, 0:9)) for y in Flux.Data.MNIST.labels(:train)],
    )
    xs_test, ys_test = (
        # convert each image into h*w*1 array of floats 
        [
            Float32.(reshape(img, 28, 28, 1)) for
            img in Flux.Data.MNIST.images(:test)
        ],
        # one-hot encode the labels
        [Float32.(Flux.onehot(y, 0:9)) for y in Flux.Data.MNIST.labels(:test)],
    )

    traindata = (xs_train, ys_train)
    testiter = DataLoader(
        shuffleobs((xs_test, ys_test)),
        eval_batch_size,
        buffered = false,
    )

    ## Model Setup
    model = get_model(
        get_config(lg, "maxiters"),
        Float32(get_config(lg, "abstol")),
        Float32(get_config(lg, "reltol")),
        get_config(lg, "model_type"),
    )

    loss_function =
        SupervisedLossContainer(Flux.Losses.logitcrossentropy, 1.0f0)

    ## Warmup
    __x = rand(28, 28, 1, 1) |> gpu
    __y = Flux.onehotbatch([1 for i in 1:1], 0:9) |> gpu
    Flux.gradient(() -> loss_function(model, __x, __y), Flux.params(model))

    nfe_counts = []
    cb = register_nfe_counts(model, nfe_counts)

    ## Training Loop
    ps = Flux.params(model)
    opt = ADAM(get_config(lg, "learning_rate"))
    step = 1
    for epoch = 1:get_config(lg, "epochs")
        try
            trainiter =
                DataLoader(shuffleobs(traindata), batch_size, buffered = false)
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
                DataLoader(traindata, eval_batch_size, buffered = false),
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
    for model_type in ["skip", "vanilla"]
        config = Dict(
            "seed" => seed,
            "learning_rate" => 0.001,
            "abstol" => 1.0f0,
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
