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


struct MnistMDEQ{has_sdeq,L1,S1,S2,D1,D2,D3,PD1,PD2,PD3,CL,C}
    layer1::L1
    scale_down1::S1
    scale_down2::S2
    deq1::D1
    deq2::D2
    deq3::D3
    post_deq1::PD1
    post_deq2::PD2
    post_deq3::PD3
    combination_layer::CL
    classifier::C
end

Flux.@functor MnistMDEQ

MnistMDEQ(has_sdeq::Bool, layers...) =
    MnistMDEQ{has_sdeq,typeof.(layers)...}(layers...)

function MnistMDEQ(layers...)
    has_sdeq = false
    for l in layers
        if l isa SkipDeepEquilibriumNetwork
            has_sdeq = true
            break
        end
    end
    return MnistMDEQ{has_sdeq,typeof.(layers)...}(layers...)
end

function (mdeq::MnistMDEQ{false})(x)
    x1 = mdeq.layer1(x)
    x2 = mdeq.scale_down1(x1)
    x3 = mdeq.scale_down2(x2)

    deq_sol_1 = mdeq.deq1(x1)
    deq_sol_2 = mdeq.deq2(x2)
    deq_sol_3 = mdeq.deq3(x3)

    post_deq_sol_1 = mdeq.post_deq1(deq_sol_1)
    post_deq_sol_2 = mdeq.post_deq2(deq_sol_2)
    post_deq_sol_3 = mdeq.post_deq3(deq_sol_3)

    x4 = mdeq.combination_layer(
        post_deq_sol_1,
        post_deq_sol_2,
        post_deq_sol_3,
    )::typeof(x)
    return mdeq.classifier(x4)
end

function (mdeq::MnistMDEQ{true})(x)
    x1 = mdeq.layer1(x)
    x2 = mdeq.scale_down1(x1)
    x3 = mdeq.scale_down2(x2)

    deq_sol_1, guess1 = mdeq.deq1(x1)::Tuple{typeof(x),typeof(x)}
    deq_sol_2, guess2 = mdeq.deq2(x2)::Tuple{typeof(x),typeof(x)}
    deq_sol_3, guess3 = mdeq.deq3(x3)::Tuple{typeof(x),typeof(x)}

    post_deq_sol_1 = mdeq.post_deq1(deq_sol_1)::typeof(x)
    post_deq_sol_2 = mdeq.post_deq2(deq_sol_2)::typeof(x)
    post_deq_sol_3 = mdeq.post_deq3(deq_sol_3)::typeof(x)

    x4 = mdeq.combination_layer(
        post_deq_sol_1,
        post_deq_sol_2,
        post_deq_sol_3,
    )::typeof(x)
    return (
        mdeq.classifier(x4),
        ((deq_sol_1, guess1), (deq_sol_2, guess2), (deq_sol_3, guess3)),
    )
end


function get_model(
    maxiters::Int,
    abstol::T,
    reltol::T,
    model_type::String,
) where {T}
    model =
        MnistMDEQ(
            Chain(
                Conv((3, 3), 1 => 16, relu; bias = true, pad = 1),  # 28 x 28 x 16
                BatchNorm(16, affine = true),
            ),
            Chain(
                Conv((4, 4), 16 => 16, relu; bias = true, pad = 1, stride = 2),  # 14 x 14 x 16,
                BatchNorm(16, affine = true),
            ),
            Chain(
                Conv((4, 4), 16 => 16, relu; bias = true, pad = 1, stride = 2),  # 7 x 7 x 16
                BatchNorm(16, affine = true),
            ),
            [
                model_type == "skip" ?
                SkipDeepEquilibriumNetwork(
                    ResNetLayer(16, 32) |> gpu,
                    ResNetLayer(16, 32) |> gpu,
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
                DeepEquilibriumNetwork(
                    ResNetLayer(16, 32) |> gpu,
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
                ) for _ = 1:3
            ]...,
            Chain(BatchNorm(16, affine = true), MaxPool((4, 4))),
            Chain(BatchNorm(16, affine = true), MaxPool((2, 2))),
            BatchNorm(16, affine = true),
            (x...) -> foldl(+, x),
            Chain(Flux.flatten, Dense(7 * 7 * 16, 10)),
        ) |> gpu
    return model
end


function (lc::SupervisedLossContainer)(model::MnistMDEQ{false}, x, y; kwargs...)
    return lc.loss_function(model(x), y)
end

function (lc::SupervisedLossContainer)(model::MnistMDEQ{true}, x, y; kwargs...)
    ŷ, ((ẑ1, z1), (ẑ2, z2), (ẑ3, z3)) = model(x)
    l1 = lc.loss_function(ŷ, y)
    l2 = mean(abs2, ẑ1 .- z1)
    l3 = mean(abs2, ẑ2 .- z2)
    l4 = mean(abs2, ẑ3 .- z3)
    return l1 + lc.λ * l2 + lc.λ * l3 + lc.λ * l4
end

FastDEQ.get_and_clear_nfe!(model::MnistMDEQ) =
    get_and_clear_nfe!.([model.deq1, model.deq2, model.deq3])


## Utilities
function register_nfe_counts(model, buffer)
    callback() = push!(buffer, get_and_clear_nfe!(model))
    return callback
end

function loss_and_accuracy(model, dataloader)
    matches, total_loss, total_datasize, total_nfe = 0, 0, 0, [0, 0, 0]
    for (x, y) in dataloader
        x = x |> gpu
        y = y |> gpu

        ŷ = model(x)
        ŷ = ŷ isa Tuple ? ŷ[1] : ŷ  # Handle SkipDEQ
        total_nfe .+= get_and_clear_nfe!(model) .* size(x, ndims(x))
        total_loss += Flux.Losses.logitcrossentropy(ŷ, y) * size(x, ndims(x))
        matches += sum(argmax.(eachcol(ŷ)) .== Flux.onecold(y |> cpu))
        total_datasize += size(x, ndims(x))
    end
    return (
        total_loss / total_datasize,
        matches / total_datasize,
        total_nfe ./ total_datasize,
    )
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
    valiter = DataLoader(valdata, batch_size, buffered = false)

    ## Model Setup
    model = get_model(
        get_config(lg, "maxiters"),
        Float32(get_config(lg, "abstol")),
        Float32(get_config(lg, "reltol")),
        get_config(lg, "model_type"),
    )

    loss_function =
        SupervisedLossContainer(Flux.Losses.logitcrossentropy, 1.0f-2)

    nfe_counts = Vector{Int64}[]
    cb = register_nfe_counts(model, nfe_counts)

    ## Training Loop
    ps = Flux.params(model)
    ### TODO: Might need LR Scheduler to get good results
    opt = ADAM(get_config(lg, "learning_rate"))
    step = 1
    for epoch = 1:get_config(lg, "epochs")
        try
            epoch_loss = 0.0f0
            epoch_nfe = Float64[0, 0, 0]
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
                        "Training/Step/NFE1" => nfe_counts[end][1],
                        "Training/Step/NFE2" => nfe_counts[end][2],
                        "Training/Step/NFE3" => nfe_counts[end][3],
                        "Training/Step/Count" => step,
                    ),
                )
                step += 1
                epoch_nfe .+= nfe_counts[end] * size(x, ndims(x))
                epoch_loss += loss * size(x, ndims(x))
            end
            ### Log the epoch loss
            epoch_loss /= size(traindata[1], ndims(traindata[1]))
            epoch_nfe ./= size(traindata[1], ndims(traindata[1]))
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
                    "Validation/Epoch/NFE1" => val_nfe[1],
                    "Validation/Epoch/NFE2" => val_nfe[2],
                    "Validation/Epoch/NFE3" => val_nfe[3],
                    "Validation/Epoch/Accuracy" => val_acc,
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
            "abstol" => 1f-1,
            "reltol" => 1f-1,
            "maxiters" => 100,
            "epochs" => 25,
            "batch_size" => 512,
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