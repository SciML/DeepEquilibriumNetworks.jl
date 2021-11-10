# Load Packages
using CUDA,
    Dates,
    DiffEqSensitivity,
    FastDEQ,
    Flux,
    FluxMPI,
    OrdinaryDiffEq,
    Statistics,
    SteadyStateDiffEq,
    MLDatasets,
    MPI,
    Plots,
    Random,
    ParameterSchedulers,
    Wandb,
    Zygote
using ParameterSchedulers: Scheduler, Cos
using MLDataPattern: splitobs, shuffleobs

MPI.Init()
CUDA.allowscalar(false)

## Models
# Resnet Layer
struct ResNetLayer{C1,C2,D1,D2,N1,N2,N3}
    conv1::C1
    conv2::C2
    dropout1::D1
    dropout2::D2
    norm1::N1
    norm2::N2
    norm3::N3
end

function Flux.gpu(r::ResNetLayer)
    return ResNetLayer(
        Flux.gpu(r.conv1),
        Flux.gpu(r.conv2),
        Flux.gpu(r.dropout1),
        Flux.gpu(r.dropout2),
        Flux.gpu(r.norm1),
        Flux.gpu(r.norm2),
        Flux.gpu(r.norm3),
    )
end

Flux.@functor ResNetLayer

function ResNetLayer(
    n_channels::Int,
    n_inner_channels::Int,
    drop1_size::Tuple,
    drop2_size::Tuple,
    dropout_rate::Real;
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
    dropout1 = VariationalHiddenDropout(dropout_rate, drop1_size)
    conv2 = Conv(
        kernel_size,
        n_inner_channels => n_channels;
        pad = kernel_size .÷ 2,
        bias = false,
        init = (dims...) -> randn(Float32, dims...) .* 0.01f0,
    )
    dropout2 = VariationalHiddenDropout(dropout_rate, drop2_size)
    norm1 = GroupNorm(n_inner_channels, num_groups, affine = affine)
    norm2 = GroupNorm(n_channels, num_groups, affine = affine)
    norm3 = GroupNorm(n_channels, num_groups, affine = affine)

    return ResNetLayer(conv1, conv2, dropout1, dropout2, norm1, norm2, norm3)
end

(rl::ResNetLayer)(z, x) = rl.norm3(
    relu.(
        z .+ rl.norm2(
            x .+ rl.dropout2(rl.conv2(rl.norm1(rl.dropout1(rl.conv1(z))))),
        ),
    ),
)

(rl::ResNetLayer)(x) = rl.norm3(
    relu.(rl.norm2(rl.dropout2(rl.conv2(rl.norm1(rl.dropout1(rl.conv1(x))))))),
)


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

function Flux.gpu(c::CIFARWidthStackedDEQ{S}) where {S}
    return CIFARWidthStackedDEQ(
        S,
        Flux.gpu(c.layer1),
        Flux.gpu(c.scale_down1),
        Flux.gpu(c.scale_down2),
        Flux.gpu(c.deq1),
        Flux.gpu(c.deq2),
        Flux.gpu(c.deq3),
        Flux.gpu(c.post_deq1),
        Flux.gpu(c.post_deq2),
        Flux.gpu(c.post_deq3),
        Flux.gpu(c.combination_layer),
        Flux.gpu(c.classifier),
    )
end

Flux.@functor CIFARWidthStackedDEQ

CIFARWidthStackedDEQ(has_sdeq::Bool, layers...) =
    CIFARWidthStackedDEQ{has_sdeq,typeof.(layers)...}(layers...)

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
    dropout_rate::Real,
    model_type::String,
) where {T}
    model = CIFARWidthStackedDEQ(
        Sequential(
            Conv((3, 3), 3 => 8, relu; bias = true, pad = 1),
            BatchNorm(8, affine = true),
            VariationalHiddenDropout(dropout_rate, (32, 32, 8, 1)),
        ),
        Sequential(
            Conv((4, 4), 8 => 16, relu; bias = true, pad = 1, stride = 2),
            BatchNorm(16, affine = true),
            VariationalHiddenDropout(dropout_rate, (16, 16, 16, 1)),
        ),
        Sequential(
            Conv((4, 4), 16 => 32, relu; bias = true, pad = 1, stride = 2),
            BatchNorm(32, affine = true),
            VariationalHiddenDropout(dropout_rate, (8, 8, 32, 1)),
        ),
        [
            model_type == "skip" ?
            SkipDeepEquilibriumNetwork(
                ResNetLayer(
                    2^(i + 2),
                    5 * (2^(i + 2)),
                    (32 ÷ (2^(i - 1)), 32 ÷ (2^(i - 1)), 5 * 2^(i + 2), 1),
                    (32 ÷ (2^(i - 1)), 32 ÷ (2^(i - 1)), 2^(i + 2), 1),
                    dropout_rate,
                ),
                ResNetLayer(
                    2^(i + 2),
                    5 * (2^(i + 2)),
                    (32 ÷ (2^(i - 1)), 32 ÷ (2^(i - 1)), 5 * 2^(i + 2), 1),
                    (32 ÷ (2^(i - 1)), 32 ÷ (2^(i - 1)), 2^(i + 2), 1),
                    dropout_rate,
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
            ) :
            DeepEquilibriumNetwork(
                ResNetLayer(
                    2^(i + 2),
                    5 * (2^(i + 2)),
                    (32 ÷ (2^(i - 1)), 32 ÷ (2^(i - 1)), 5 * 2^(i + 2), 1),
                    (32 ÷ (2^(i - 1)), 32 ÷ (2^(i - 1)), 2^(i + 2), 1),
                    dropout_rate,
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
            ) for i = 1:3
        ]...,
        Sequential(
            BatchNorm(8, affine = true),
            Conv((4, 4), 8 => 16, relu; bias = true, pad = 1, stride = 2),
            VariationalHiddenDropout(dropout_rate, (16, 16, 16, 1)),
            BatchNorm(16, affine = true),
            Conv((4, 4), 16 => 32, relu; bias = true, pad = 1, stride = 2),
            VariationalHiddenDropout(dropout_rate, (8, 8, 32, 1)),
        ),
        Sequential(
            BatchNorm(16, affine = true),
            Conv((4, 4), 16 => 32, relu; bias = true, pad = 1, stride = 2),
            VariationalHiddenDropout(dropout_rate, (8, 8, 32, 1)),
        ),
        Sequential(
            BatchNorm(32, affine = true),
            VariationalHiddenDropout(dropout_rate, (8, 8, 32, 1)),
        ),
        (x...) -> foldl(+, x),
        Sequential(Flux.flatten, Dense(8 * 8 * 32, 10)),
    )
    return DataParallelFluxModel(
        model,
        [i % length(CUDA.devices()) for i = 1:MPI.Comm_size(MPI.COMM_WORLD)],
    )
end

(lc::SupervisedLossContainer)(model::DataParallelFluxModel, x, y; kwargs...) =
    lc(model.model, x, y, kwargs...)

function (lc::SupervisedLossContainer)(
    model::CIFARWidthStackedDEQ{false},
    x,
    y;
    kwargs...,
)
    return lc.loss_function(model(x), y)
end

function (lc::SupervisedLossContainer)(
    model::CIFARWidthStackedDEQ{true},
    x,
    y;
    kwargs...,
)
    ŷ, ((ẑ1, z1), (ẑ2, z2), (ẑ3, z3)) = model(x)
    l1 = lc.loss_function(ŷ, y)
    l2 = mean(abs2, ẑ1 .- z1)
    l3 = mean(abs2, ẑ2 .- z2)
    l4 = mean(abs2, ẑ3 .- z3)
    return l1 + lc.λ * l2 + lc.λ * l3 + lc.λ * l4
end

FastDEQ.get_and_clear_nfe!(model::CIFARWidthStackedDEQ) =
    get_and_clear_nfe!.([model.deq1, model.deq2, model.deq3])

FastDEQ.get_and_clear_nfe!(model::DataParallelFluxModel) =
    get_and_clear_nfe!(model.model)


## Utilities
function register_nfe_counts(model, buffer)
    callback() = push!(buffer, get_and_clear_nfe!(model))
    return callback
end

function loss_and_accuracy(model, dataloader)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    rank != 0 && return (0, 0, [0, 0, 0])

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
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    ## Setup Logging & Experiment Configuration
    lg = WandbLoggerMPI(
        project = "FastDEQ.jl",
        name = "fastdeqjl-supervised_cifar10_classication-$(now())",
        config = config,
    )

    ## Reproducibility
    Random.seed!(get_config(lg, "seed"))

    ## Model Setup
    model = get_model(
        get_config(lg, "maxiters"),
        Float32(get_config(lg, "abstol")),
        Float32(get_config(lg, "reltol")),
        Float64(get_config(lg, "dropout_rate")),
        get_config(lg, "model_type"),
    )

    ## Dataset
    batch_size = get_config(lg, "batch_size")
    eval_batch_size = get_config(lg, "eval_batch_size")

    _xs_train, _ys_train = CIFAR10.traindata(Float32)
    _xs_test, _ys_test = CIFAR10.testdata(Float32)

    xs_train, ys_train = _xs_train, Float32.(Flux.onehotbatch(_ys_train, 0:9))
    xs_test, ys_test = _xs_test, Float32.(Flux.onehotbatch(_ys_test, 0:9))

    traindata = (xs_train, ys_train)
    trainiter = DataParallelDataLoader(
        traindata;
        batchsize = batch_size,
        shuffle = true,
    )
    testiter = DataParallelDataLoader(
        (xs_test, ys_test);
        batchsize = eval_batch_size,
        shuffle = false,
    )

    ## Loss Function
    loss_function =
        SupervisedLossContainer(Flux.Losses.logitcrossentropy, 1.0f0)

    nfe_counts = Vector{Int64}[]
    cb = register_nfe_counts(model, nfe_counts)

    ## Warmup with a smaller batch
    _x_warmup, _y_warmup =
        rand(32, 32, 3, 1) |> gpu, Flux.onehotbatch([1], 0:9) |> gpu
    Zygote.gradient(
        () -> loss_function(model, _x_warmup, _y_warmup),
        Flux.params(model),
    )
    @info "Rank $rank: Warmup Completed"

    ## Training Loop
    ps = Flux.params(model)
    opt = Scheduler(
        Cos(
            get_config(lg, "learning_rate"),
            get_config(lg, "learning_rate") / 10,
            100
        ),
        ADAM(
            get_config(lg, "learning_rate"),
            (0.9, 0.999)
        )
    )
    step = 1
    train_vec = zeros(Float32, 5)
    test_vec = zeros(Float32, 5)

    datacount_trainiter = length(trainiter.indices)
    datacount_testiter = length(testiter.indices)
    datacount_trainiter_total = size(xs_train, ndims(xs_train))
    datacount_testiter_total = size(xs_test, ndims(xs_test))

    for epoch = 1:get_config(lg, "epochs")
        try
            for (x, y) in trainiter
                x = x |> gpu
                y = y |> gpu

                _res = Zygote.withgradient(() -> loss_function(model, x, y), ps)
                loss = _res.val
                gs = _res.grad

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
            train_loss, train_acc, train_nfe =
                loss_and_accuracy(model, trainiter)

            train_vec[0] .= train_loss * datacount_trainiter
            train_vec[1] .= train_acc * datacount_trainiter
            train_vec[2:end] .= train_nfe .* datacount_trainiter
            safe_reduce!(train_vec, +, 0, comm)
            train_loss, train_acc, train_nfe = (
                train_vec[0] / datacount_trainiter_total,
                train_vec[1] / datacount_trainiter_total,
                train_vec[2:end] ./ datacount_trainiter_total
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

            test_vec[0] .= test_loss * datacount_testiter
            test_vec[1] .= test_acc * datacount_testiter
            test_vec[2:end] .= test_nfe .* datacount_testiter
            safe_reduce!(test_vec, +, 0, comm)
            test_loss, test_acc, test_nfe = (
                test_vec[0] / datacount_testiter_total,
                test_vec[1] / datacount_testiter_total,
                test_vec[2:end] ./ datacount_testiter_total
            )

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

    lg !== nothing && close(lg)

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
            "abstol" => 1f-3,
            "reltol" => 1f-3,
            "maxiters" => 20,
            "epochs" => 50,
            "dropout_rate" => 0.1,
            "batch_size" => 32,
            "eval_batch_size" => 64,
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
