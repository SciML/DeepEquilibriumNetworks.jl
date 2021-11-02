# Load Packages
using CUDA,
    Dates,
    DiffEqSensitivity,
    FastDEQ,
    Flux,
    OrdinaryDiffEq,
    Statistics,
    SteadyStateDiffEq,
    MLDatasets,
    Plots,
    Random,
    Wandb,
    Zygote
using DataLoaders: DataLoader
using MLDataPattern: splitobs, shuffleobs

CUDA.allowscalar(false)

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


struct CIFARWidthStackedDEQ{has_sdeq,L1,S1,S2,D1,D2,D3,PD1,PD2,PD3,CL,C}
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

Flux.@functor CIFARWidthStackedDEQ

CIFARWidthStackedDEQ(has_sdeq::Bool, layers...) =
    MnistWidthStackedDEQ{has_sdeq,typeof.(layers)...}(layers...)

function CIFARWidthStackedDEQ(layers...)
    has_sdeq = false
    for l in layers
        if l isa SkipDeepEquilibriumNetwork
            has_sdeq = true
            break
        end
    end
    return CIFARWidthStackedDEQ{has_sdeq,typeof.(layers)...}(layers...)
end

function (mdeq::CIFARWidthStackedDEQ{false})(x)
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

function (mdeq::CIFARWidthStackedDEQ{true})(x)
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
        CIFARWidthStackedDEQ(
            Chain(
                Conv((3, 3), 3 => 16, relu; bias = true, pad = 1),
                BatchNorm(16, affine = true),
            ),
            Chain(
                Conv((4, 4), 16 => 16, relu; bias = true, pad = 1, stride = 2),
                BatchNorm(16, affine = true),
            ),
            Chain(
                Conv((4, 4), 16 => 16, relu; bias = true, pad = 1, stride = 2),
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
            Chain(Flux.flatten, Dense(8 * 8 * 16, 10)),
        ) |> gpu
    return model
end


function (lc::SupervisedLossContainer)(model::CIFARWidthStackedDEQ{false}, x, y; kwargs...)
    return lc.loss_function(model(x), y)
end

function (lc::SupervisedLossContainer)(model::CIFARWidthStackedDEQ{true}, x, y; kwargs...)
    ŷ, ((ẑ1, z1), (ẑ2, z2), (ẑ3, z3)) = model(x)
    l1 = lc.loss_function(ŷ, y)
    l2 = mean(abs2, ẑ1 .- z1)
    l3 = mean(abs2, ẑ2 .- z2)
    l4 = mean(abs2, ẑ3 .- z3)
    return l1 + lc.λ * l2 + lc.λ * l3 + lc.λ * l4
end

FastDEQ.get_and_clear_nfe!(model::CIFARWidthStackedDEQ) =
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
        name = "fastdeqjl-supervised_cifar10_classication-$(now())",
        config = config,
    )

    ## Reproducibility
    Random.seed!(get_config(lg, "seed"))

    ## Dataset
    batch_size = get_config(lg, "batch_size")
    eval_batch_size = get_config(lg, "eval_batch_size")

    _xs_train, _ys_train = CIFAR10.traindata(Float32)
    _xs_test, _ys_test = CIFAR10.testdata(Float32)

    xs_train, ys_train = Flux.unbatch(_xs_train), Float32.(Flux.onehotbatch(_ys_train, 0:9))
    xs_test, ys_test = Flux.unbatch(_xs_test), Float32.(Flux.onehotbatch(_ys_test, 0:9))

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

    nfe_counts = Vector{Int64}[]
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
                        "Training/Step/NFE1" => nfe_counts[end][1],
                        "Training/Step/NFE2" => nfe_counts[end][2],
                        "Training/Step/NFE3" => nfe_counts[end][3],
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
                    "Training/Epoch/NFE1" => train_nfe[1],
                    "Training/Epoch/NFE2" => train_nfe[2],
                    "Training/Epoch/NFE3" => train_nfe[3],
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
                    "Testing/Epoch/NFE1" => test_nfe[1],
                    "Testing/Epoch/NFE2" => test_nfe[2],
                    "Testing/Epoch/NFE3" => test_nfe[3],
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
            "abstol" => 1f-1,
            "reltol" => 1f-1,
            "maxiters" => 40,
            "epochs" => 25,
            "batch_size" => 512,
            "eval_batch_size" => 2048,
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
