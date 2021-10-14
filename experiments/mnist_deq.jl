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
    n_inner_channels::Int,
    kernel_size::Tuple{Int,Int} = (3, 3),
    num_groups::Int = 8,
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
    norm1 = GroupNorm(n_inner_channels, num_groups, affine = false)
    norm2 = GroupNorm(n_channels, num_groups, affine = false)
    norm3 = GroupNorm(n_channels, num_groups, affine = false)

    return ResNetLayer(conv1, conv2, norm1, norm2, norm3)
end

(rl::ResNetLayer)(z, x) =
    rl.norm3(relu.(z .+ rl.norm2(x .+ rl.conv2(rl.norm1(rl.conv1(z))))))

function get_model(
    maxiters::Int,
    abstol::T,
    reltol::T,
    model_type::String,
) where {T}
    if model_type == "vanilla"
        model =
            DEQChain(
                Conv((3, 3), 1 => 48, relu; bias = true, pad = 1),  # 28 x 28 x 48
                BatchNorm(48, affine = true),
                DeepEquilibriumNetwork(
                    ResNetLayer(48, 64) |> gpu,
                    DynamicSS(Tsit5(); abstol = abstol, reltol = reltol),
                    maxiters = maxiters,
                    sensealg = SteadyStateAdjoint(
                        autodiff = false,
                        autojacvec = ZygoteVJP(),
                        linsolve = LinSolveKrylovJL(
                            rtol = reltol,
                            atol = abstol,
                            itmax = maxiters,
                        ),
                    ),
                ),
                BatchNorm(48, affine = true),
                MeanPool((8, 8)),  # 3 x 3 x 48
                Flux.flatten,
                Dense(3 * 3 * 48, 10),
            ) |> gpu
    elseif model_type == "skip"
        model =
            DEQChain(
                Conv((3, 3), 1 => 48, relu; bias = true, pad = 1),  # 28 x 28 x 48
                BatchNorm(48, affine = true),
                SkipDeepEquilibriumNetwork(
                    ResNetLayer(48, 64) |> gpu,
                    Chain(
                        Conv((3, 3), 48 => 64, relu; bias = true, pad = 1),
                        GroupNorm(64, 8, affine = false),
                        Conv((3, 3), 64 => 48, relu; bias = true, pad = 1)
                    ) |> gpu,
                    DynamicSS(Tsit5(); abstol = abstol, reltol = reltol),
                    maxiters = maxiters,
                    sensealg = SteadyStateAdjoint(
                        autodiff = false,
                        autojacvec = ZygoteVJP(),
                        linsolve = LinSolveKrylovJL(
                            rtol = reltol,
                            atol = abstol,
                            itmax = maxiters,
                        ),
                    ),
                ),
                BatchNorm(48, affine = true),
                MeanPool((8, 8)),  # 3 x 3 x 48
                Flux.flatten,
                Dense(3 * 3 * 48, 10),
            ) |> gpu
    else
        throw(ArgumentError("$model_type must be either `vanilla` or `skip`"))
    end
    return model
end


## Utilities
function register_nfe_counts(model, buffer)
    callback() = push!(buffer, get_and_clear_nfe!(model))
    return callback
end

function loss_and_accuracy(model, dataloader)
    matches, total_loss, total_datasize, total_nfe = 0, 0, 0, 0
    iter = ProgressBar(dataloader)
    for (x, y) in iter
        x = x |> gpu
        y = y |> gpu

        ŷ = model(x)
        total_nfe += get_and_clear_nfe!(model) * size(x, ndims(x))
        total_loss += lossfn(ŷ, y) * size(x, ndims(x))
        matches += sum(argmax.(eachcol(ŷ)) .== Flux.onecold(y |> cpu))
        total_datasize += size(x, ndims(x))
    end
    return total_loss / total_datasize, matches / total_datasize, total_nfe / total_datasize
end


## Training Function
function train(config::Dict)
    ## Setup Logging & Experiment Configuration
    lg = WandbLogger(
        project = "FastDEQ.jl",
        name = "fastdeqjl-supervised_mnist_classication-$(now())",
        config = config,
    )

    ## Reproducibility
    Random.seed!(get_config(lg, "seed"))

    ## Dataset
    batch_size = get_config(lg, "batch_size")
    xs, ys = (
        # convert each image into h*w*1 array of floats 
        [Float32.(reshape(img, 28, 28, 1)) for img in Flux.Data.MNIST.images()],
        # one-hot encode the labels
        [Float32.(Flux.onehot(y, 0:9)) for y in Flux.Data.MNIST.labels()],
    )

    # split into training and validation sets
    traindata, valdata = splitobs((xs, ys), at = 0.9)

    # create iterators
    valiter = DataLoader(valdata, batch_size, buffered = false);

    ## Model Setup
    model = get_model(
        get_config(lg, "maxiters"),
        get_config(lg, "abstol"),
        get_config(lg, "reltol"),
        get_config(lg, "model_type"),
    )

    loss_function = SupervisedLossContainer(Flux.Losses.logitcrossentropy, 1f-2)

    nfe_counts = Int64[]
    cb = register_nfe_counts(model, nfe_counts)

    ## Training Loop
    ps = Flux.params(model)
    ### TODO: Might need LR Scheduler to get good results
    opt = ADAM(get_config(lg, "learning_rate"))
    step = 1
    for epoch = 1:get_config(lg, "epochs")
        try
            epoch_loss = 0.0f0
            epoch_nfe = 0
            trainiter = DataLoader(shuffleobs(traindata), batch_size, buffered = false)
            for (x, y) in trainiter
                x = x |> gpu
                y = y |> gpu

                loss, back = Zygote.pullback(() -> loss_function(model, x, y), ps)
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
                epoch_nfe += nfe_counts[end] * size(x, ndims(x))
                epoch_loss += loss * size(x, ndims(x))
            end
            ### Log the epoch loss
            epoch_loss /= size(x_data, ndims(x_data))
            epoch_nfe /= size(x_data, ndims(x_data))
            log(
                lg,
                Dict(
                    "Training/Epoch/Loss" => epoch_loss,
                    "Training/Epoch/NFE" => epoch_nfe,
                    "Training/Epoch/Count" => epoch,
                ),
            )

            ### Validation Loss/Accuracy
            val_loss, val_acc, val_nfe = loss_and_accuracy(model, valiter)
            log(
                lg,
                Dict(
                    "Validation/Epoch/Loss" => val_loss,
                    "Validation/Epoch/NFE" => val_nfe,
                    "Validation/Epoch/Accuracy" => val_acc,
                )
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
        config = Dict("seed" => seed,
                      "learning_rate" => 0.001,
                      "abstol" => 1f-2,
                      "reltol" => 1f-2,
                      "maxiters" => 10,
                      "epochs" => 10,
                      "batch_size" => 512,
                      "model_type" => model_type)

        model, nfe_counts = train(config)

        push!(nfe_count_dict[model_type], nfe_counts)
    end
end

plot_nfe_counts(
    vec(mean(hcat(nfe_count_dict["vanilla"]...), dims = 2)),
    vec(mean(hcat(nfe_count_dict["skip"]...), dims = 2)),
)