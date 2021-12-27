# Load Packages
using Dates, FastDEQ, MLDatasets, MPI, Serialization, Statistics, ParameterSchedulers, Plots, Random, Wandb
using ParameterSchedulers: Stateful, Scheduler, Cos
using MLDataPattern: splitobs, shuffleobs

CUDA.versioninfo()

FluxMPI.Init()
enable_fast_mode!()
CUDA.allowscalar(false)

const MPI_COMM_WORLD = MPI.COMM_WORLD
const MPI_COMM_SIZE = MPI.Comm_size(MPI_COMM_WORLD)

## Models
function get_model(maxiters::Int, abstol::T, reltol::T, dropout_rate::Real, model_type::String, batch_size::Int,
                   solver_type::String, args...; kwargs...) where {T}
    layer_kwargs = Dict(:norm_layer => GroupNormV2, :group_count => 8,
                        :conv_kwargs => Dict{Symbol,Any}(:bias => false, :init => normal_init()),
                        :norm_kwargs => Dict{Symbol,Any}(:affine => true, :track_stats => false))

    initial_layers = Chain(conv3x3_norm(3 => 24, gelu; norm_layer=BatchNormV2,
                                        norm_kwargs=Dict{Symbol,Any}(:track_stats => true, :affine => true),
                                        conv_kwargs=Dict{Symbol,Any}(:init => normal_init(),
                                                                     :bias => normal_init()(24))).layers...,
                           conv3x3_norm(24 => 24, gelu; norm_layer=BatchNormV2,
                                        norm_kwargs=Dict{Symbol,Any}(:track_stats => true, :affine => true),
                                        conv_kwargs=Dict{Symbol,Any}(:init => normal_init(),
                                                                     :bias => normal_init()(24))).layers...)

    main_layers = (BasicResidualBlock((32, 32), 24, 24; dropout_rate=dropout_rate, num_gn_groups=8),
                   BasicResidualBlock((16, 16), 24, 24; dropout_rate=dropout_rate, num_gn_groups=8))

    mapping_layers = [NoOpLayer() downsample_module(24 => 24, 32 => 16, gelu; layer_kwargs...);
                      upsample_module(24 => 24, 16 => 32, gelu; layer_kwargs...) NoOpLayer()]

    post_fuse_layers = (Chain(x -> gelu.(x), conv1x1(24 => 24; bias=false, init=normal_init()),
                              GroupNormV2(24, 4; affine=true, track_stats=false)),
                        Chain(x -> gelu.(x), conv1x1(24 => 24; bias=false, init=normal_init()),
                              GroupNormV2(24, 4; affine=true, track_stats=false)))

    final_layers = Chain(Parallel(+,
                                  Chain(BasicBottleneckBlock(24 => 8),
                                        conv3x3(8 * 4 => 16 * 4; stride=2, bias=normal_init()(16 * 4),
                                                init=normal_init()),
                                        BatchNormV2(16 * 4, gelu; track_stats=true, affine=true)),
                                  BasicBottleneckBlock(24 => 16)),
                         conv1x1_norm(16 * 4 => 200, gelu; norm_layer=BatchNormV2,
                                      norm_kwargs=Dict{Symbol,Any}(:track_stats => true, :affine => true),
                                      conv_kwargs=Dict{Symbol,Any}(:init => normal_init(), :bias => normal_init()(200))).layers...,
                         GlobalMeanPool(), FlattenLayer(), Dense(200, 10))

    if solver_type == "dynamicss"
        solver = get_default_dynamicss_solver(reltol, abstol, Tsit5(); mode=:rel_deq_best)
    else
        solver = get_default_ssrootfind_solver(reltol, abstol, LimitedMemoryBroydenSolver; device=gpu,
                                               original_dims=(32 * 32 * 24 + 16 * 16 * 24, 1), batch_size=batch_size,
                                               maxiters=maxiters)
    end

    if model_type == "skip"
        deq = MultiScaleSkipDeepEquilibriumNetwork(main_layers, mapping_layers,
                                                   (BasicResidualBlock((32, 32), 24, 24; num_gn_groups=8),
                                                    downsample_module(24 => 24, 32 => 16; layer_kwargs...)), solver;
                                                   post_fuse_layers=post_fuse_layers, maxiters=maxiters,
                                                   sensealg=get_default_ssadjoint(reltol, abstol, min(maxiters, 15)),
                                                   verbose=false)
    else
        _deq = model_type == "vanilla" ? MultiScaleDeepEquilibriumNetwork : MultiScaleSkipDeepEquilibriumNetwork
        deq = _deq(main_layers, mapping_layers, solver; post_fuse_layers=post_fuse_layers, maxiters=maxiters,
                   sensealg=get_default_ssadjoint(reltol, abstol, min(maxiters, 15)), verbose=false)
    end
    model = DEQChain(initial_layers, deq, final_layers)

    return (MPI_COMM_SIZE > 1 ? DataParallelFluxModel : gpu)(model)
end

## Utilities
register_nfe_counts(deq, buffer) = () -> push!(buffer, get_and_clear_nfe!(deq))

function invoke_gc()
    GC.gc(true)
    CUDA.reclaim()
    return nothing
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
function train(config::Dict, name_extension::String="")
    comm = MPI_COMM_WORLD
    rank = MPI.Comm_rank(comm)

    ## Setup Logging & Experiment Configuration
    t = MPI.Bcast!([now()], 0, comm)[1]
    expt_name = "fastdeqjl-supervised_cifar10_classification-$(t)-$(name_extension)"
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

    ### CIFAR 10 Normalization Stats
    μ = reshape([0.4914, 0.4822, 0.4465], 1, 1, :, 1)
    σ² = reshape([0.2023, 0.1994, 0.2010], 1, 1, :, 1)

    _xs_train, _ys_train = CIFAR10.traindata(Float32)
    _xs_train = (_xs_train .- μ) ./ σ²
    _xs_test, _ys_test = CIFAR10.testdata(Float32)
    _xs_test = (_xs_test .- μ) ./ σ²

    xs_train, ys_train = _xs_train, Float32.(Flux.onehotbatch(_ys_train, 0:9))
    xs_test, ys_test = _xs_test, Float32.(Flux.onehotbatch(_ys_test, 0:9))

    traindata = (xs_train, ys_train)
    trainiter = DataParallelDataLoader(traindata; batchsize=batch_size, shuffle=true)
    testiter = DataParallelDataLoader((xs_test, ys_test); batchsize=eval_batch_size, shuffle=false)

    ## Loss Function
    loss_function = SupervisedLossContainer(Flux.Losses.logitcrossentropy, 1.0f-2, 1.0f0)

    nfe_counts = []
    cb = register_nfe_counts(model, nfe_counts)

    ## Warmup with a smaller batch
    _x_warmup, _y_warmup = gpu(rand(32, 32, 3, 1)), gpu(Flux.onehotbatch([1], 0:9))
    loss_function(model, _x_warmup, _y_warmup)
    @info "Rank $rank: Forward Pass Warmup Completed"
    Zygote.gradient(() -> loss_function(model, _x_warmup, _y_warmup), Flux.params(model))
    @info "Rank $rank: Warmup Completed"
    get_and_clear_nfe!(model)

    ## Training Loop
    ps = Flux.params(model)

    sched = Stateful(Cos(get_config(lg_wandb, "learning_rate"), 1e-6,
                         length(trainiter) * get_config(lg_wandb, "epochs")))
    opt = ADAMW(get_config(lg_wandb, "learning_rate"), (0.9, 0.999), get_config(lg_wandb, "weight_decay"))

    watch = ParameterStateGradientWatcher(model.model, :mdeq)

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
            epoch_end_time = 0
            invoke_gc()
            for (x, y) in trainiter
                x = gpu(x)
                y = gpu(y)

                start_time = time()

                _res = Zygote.withgradient(() -> loss_function(model, x, y), ps)
                loss = _res.val
                gs = _res.grad

                Flux.Optimise.update!(opt, ps, gs)
                opt[3].eta = ParameterSchedulers.next!(sched)

                epoch_end_time += time() - start_time

                ### Store the NFE Count
                cb()

                ### Log the losses
                log(lg_wandb,
                    Dict("Training/Step/Loss" => loss, "Training/Step/NFE" => nfe_counts[end],
                         "Training/Step/Count" => step))

                # This is pretty expensive so do after every 50 steps
                # NOTE: This is mainly for debugging purposes so if you suspect something is
                #       wrong with the model you can uncomment this
                # step % 50 == 1 && log(lg_wandb, watch, gs)

                lg_term(; records=Dict("Train/Running/NFE" => nfe_counts[end], "Train/Running/Loss" => loss))
                step += 1
            end

            ### Training Loss/Accuracy
            invoke_gc()
            train_loss, train_acc, train_nfe = loss_and_accuracy(model, trainiter)

            if MPI_COMM_SIZE > 1
                train_vec .= [train_loss, train_acc, train_nfe] .* datacount_trainiter
                safe_reduce!(train_vec, +, 0, comm)
                train_loss, train_acc, train_nfe = train_vec ./ datacount_trainiter_total
            end

            log(lg_wandb,
                Dict("Training/Epoch/Count" => epoch, "Training/Epoch/Loss" => train_loss,
                     "Training/Epoch/NFE" => train_nfe, "Training/Epoch/Accuracy" => train_acc,
                     "Training/Epoch/Time" => epoch_end_time))

            ### Testing Loss/Accuracy
            invoke_gc()
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
                     "Testing/Epoch/Accuracy" => test_acc, "Testing/Epoch/Time" => test_end_time - test_start_time))
            lg_term(epoch, train_nfe, train_acc, train_loss, epoch_end_time, test_nfe, test_acc, test_loss,
                    test_end_time - test_start_time)

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
experiment_configurations = []
for seed in [6171, 3859, 2961]  # Generated this by randomly sampling from 1:10000
    for solver_type in ["dynamicss", "ssrootfind"]
        for model_type in ["skip", "vanilla"]
            push!(experiment_configurations, (seed, model_type, solver_type))
        end
    end
end

TASK_ID = parse(Int, ARGS[1])
NUM_TASKS = parse(Int, ARGS[2])

for i in TASK_ID:NUM_TASKS:length(experiment_configurations)
    (seed, model_type, solver_type) = experiment_configurations[i]

    if MPI.Comm_rank(MPI_COMM_WORLD) == 0
        @info "Seed = $seed | Model Type = $model_type | Solver Type = $solver_type"
    end

    config = Dict("seed" => seed, "learning_rate" => 0.001, "abstol" => 5.0f-2, "reltol" => 5.0f-2,
                  "maxiters" => 50, "epochs" => 50, "dropout_rate" => 0.25, "batch_size" => 64,
                  "eval_batch_size" => 64, "model_type" => model_type, "solver_type" => solver_type,
                  "weight_decay" => 0.0000025)

    model, nfe_counts = train(config, "seed-$(seed)_model-$(model_type)_solver-$(solver_type)")
end
