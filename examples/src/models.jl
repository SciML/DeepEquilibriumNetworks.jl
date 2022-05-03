# Building Blocks
## Helpful Functional Wrappers
function conv1x1(mapping, activation=identity; stride::Int=1, bias=false, kwargs...)
    return Conv((1, 1), mapping, activation; stride=stride, pad=0, bias=bias, initW=NormalInitializer(), kwargs...)
end

function conv3x3(mapping, activation=identity; stride::Int=1, bias=false, kwargs...)
    return Conv((3, 3), mapping, activation; stride=stride, pad=1, bias=bias, initW=NormalInitializer(), kwargs...)
end

function conv5x5(mapping, activation=identity; stride::Int=1, bias=false, kwargs...)
    return Conv((5, 5), mapping, activation; stride=stride, pad=2, bias=bias, initW=NormalInitializer(), kwargs...)
end

reassociate(x::NTuple{2,<:AbstractArray}, y) = (x[1], (x[2], y))

## Downsample Module
function downsample_module(mapping, level_diff, activation; group_count=8)
    in_channels, out_channels = mapping

    function intermediate_mapping(i)
        if in_channels * (2^level_diff) == out_channels
            return (in_channels * (2^(i - 1))) => (in_channels * (2^i))
        else
            return i == level_diff ? in_channels => out_channels : in_channels => in_channels
        end
    end

    layers = Lux.AbstractExplicitLayer[]
    for i in 1:level_diff
        inchs, outchs = intermediate_mapping(i)
        push!(layers, conv3x3(inchs => outchs; stride=2))
        # push!(layers, GroupNorm(outchs, group_count, activation; affine=true, track_stats=false))
        push!(layers, BatchNorm(outchs, activation; affine=true, track_stats=false))
    end
    return Chain(layers...)
end

## Upsample Module
function upsample_module(mapping, level_diff, activation; upsample_mode::Symbol=:nearest, group_count=8)
    in_channels, out_channels = mapping

    function intermediate_mapping(i)
        if out_channels * (2^level_diff) == in_channels
            (in_channels ÷ (2^(i - 1))) => (in_channels ÷ (2^i))
        else
            i == level_diff ? in_channels => out_channels : in_channels => in_channels
        end
    end

    layers = Lux.AbstractExplicitLayer[]
    for i in 1:level_diff
        inchs, outchs = intermediate_mapping(i)
        push!(layers, conv1x1(inchs => outchs))
        # push!(layers, GroupNorm(outchs, group_count, activation; affine=true, track_stats=false))
        push!(layers, BatchNorm(outchs, activation; affine=true, track_stats=false))
        push!(layers, Upsample(upsample_mode; scale=2))
    end
    return Chain(layers...)
end

## Residual Block
function ResidualBlockV1(
    mapping;
    deq_expand::Int=5,
    num_gn_groups::Int=4,
    downsample=NoOpLayer(),
    n_big_kernels::Int=0,
    dropout_rate::Real=0.0f0,
    gn_affine::Bool=true,
    weight_norm::Bool=true,
    gn_track_stats::Bool=false,
)
    inplanes, outplanes = mapping
    inner_planes = outplanes * deq_expand
    conv1 = (n_big_kernels >= 1 ? conv5x5 : conv3x3)(inplanes => inner_planes; bias=false)
    conv2 = (n_big_kernels >= 2 ? conv5x5 : conv3x3)(inner_planes => outplanes; bias=false)

    conv1, conv2 = if weight_norm
        WeightNorm(conv1, (:weight,), (4,)), WeightNorm(conv2, (:weight,), (4,))
    else
        conv1, conv2
    end

    # gn1 = GroupNorm(inner_planes, num_gn_groups, relu; affine=gn_affine, track_stats=gn_track_stats)
    # gn2 = GroupNorm(outplanes, num_gn_groups, relu; affine=gn_affine, track_stats=gn_track_stats)
    # gn3 = GroupNorm(outplanes, num_gn_groups; affine=gn_affine, track_stats=gn_track_stats)
    gn1 = BatchNorm(inner_planes, relu; affine=gn_affine, track_stats=gn_track_stats)
    gn2 = BatchNorm(outplanes, relu; affine=gn_affine, track_stats=gn_track_stats)
    gn3 = BatchNorm(outplanes; affine=gn_affine, track_stats=gn_track_stats)

    dropout = iszero(dropout_rate) ? NoOpLayer() : VariationalHiddenDropout(dropout_rate)

    return Chain(
        Parallel(
            reassociate, # Reassociate and Merge
            Chain(conv1, gn1, conv2, BranchLayer(downsample, dropout)),  # For x
            NoOpLayer(),  # For injection
        ),
        Parallel(
            +,
            NoOpLayer(),  # For y1
            Chain(
                WrappedFunction(y2i -> y2i[1] .+ y2i[2]),  # Since injection could be a scalar
                gn2,
            ),  # For (y2, injection)
        ),
        ActivationFunction(relu),
        gn3,
    )
end

function ResidualBlockV2(
    mapping;
    deq_expand::Int=5,
    num_gn_groups::Int=4,
    downsample=NoOpLayer(),
    n_big_kernels::Int=0,
    dropout_rate::Real=0.0f0,
    gn_affine::Bool=true,
    weight_norm::Bool=true,
    gn_track_stats::Bool=false,
)
    inplanes, outplanes = mapping
    inner_planes = outplanes * deq_expand
    conv1 = (n_big_kernels >= 1 ? conv5x5 : conv3x3)(inplanes => inner_planes; bias=false)
    conv2 = (n_big_kernels >= 2 ? conv5x5 : conv3x3)(inner_planes => outplanes; bias=false)

    conv1, conv2 = if weight_norm
        WeightNorm(conv1, (:weight,), (4,)), WeightNorm(conv2, (:weight,), (4,))
    else
        conv1, conv2
    end

    # gn1 = GroupNorm(inner_planes, num_gn_groups, relu; affine=gn_affine, track_stats=gn_track_stats)
    # gn2 = GroupNorm(outplanes, num_gn_groups, relu; affine=gn_affine, track_stats=gn_track_stats)
    # gn3 = GroupNorm(outplanes, num_gn_groups; affine=gn_affine, track_stats=gn_track_stats)
    gn1 = BatchNorm(inner_planes, relu; affine=gn_affine, track_stats=gn_track_stats)
    gn2 = BatchNorm(outplanes, relu; affine=gn_affine, track_stats=gn_track_stats)
    gn3 = BatchNorm(outplanes; affine=gn_affine, track_stats=gn_track_stats)

    dropout = iszero(dropout_rate) ? NoOpLayer() : VariationalHiddenDropout(dropout_rate)

    return Chain(
        conv1,
        gn1,
        conv2,
        Parallel(+, downsample, Chain(dropout, gn2)),
        # WrappedFunction(Base.Fix1(broadcast, relu)),
        ActivationFunction(relu),
        gn3,
    )
end

function BottleneckBlockV1(mapping::Pair, expansion::Int=4; bn_track_stats::Bool=false, bn_affine::Bool=true)
    rescale = if first(mapping) != last(mapping) * expansion
        Chain(
            conv1x1(first(mapping) => last(mapping) * expansion),
            BatchNorm(last(mapping) * expansion; track_stats=bn_track_stats, affine=bn_affine),
        )
    else
        NoOpLayer()
    end

    return Chain(
        Parallel(reassociate, BranchLayer(rescale, conv1x1(mapping)), NoOpLayer()),
        Parallel(
            +,
            NoOpLayer(),
            Chain(
                WrappedFunction(y2i -> y2i[1] .+ y2i[2]),  # Since injection could be a scalar
                Chain(
                    BatchNorm(last(mapping), relu; affine=bn_affine, track_stats=bn_track_stats),
                    conv3x3(last(mapping) => last(mapping) * expansion),
                    BatchNorm(last(mapping) * expansion, relu; track_stats=bn_track_stats, affine=bn_affine),
                    conv1x1(last(mapping) * expansion => last(mapping) * expansion),
                    BatchNorm(last(mapping) * expansion; track_stats=bn_track_stats, affine=bn_affine),
                ),
            ),
        ),
        ActivationFunction(relu),
    )
end

function BottleneckBlockV2(mapping::Pair, expansion::Int=4; bn_track_stats::Bool=false, bn_affine::Bool=true)
    rescale = if first(mapping) != last(mapping) * expansion
        Chain(
            conv1x1(first(mapping) => last(mapping) * expansion),
            BatchNorm(last(mapping) * expansion; track_stats=bn_track_stats, affine=bn_affine),
        )
    else
        NoOpLayer()
    end

    return Chain(
        Parallel(
            +,
            rescale,
            Chain(
                conv1x1(mapping),
                BatchNorm(last(mapping), relu; affine=bn_affine, track_stats=bn_track_stats),
                conv3x3(last(mapping) => last(mapping) * expansion),
                BatchNorm(last(mapping) * expansion, relu; track_stats=bn_track_stats, affine=bn_affine),
                conv1x1(last(mapping) * expansion => last(mapping) * expansion),
                BatchNorm(last(mapping) * expansion; track_stats=bn_track_stats, affine=bn_affine),
            ),
        ),
        ActivationFunction(relu),
    )
end

# Dataset Specific Models
get_model(econfig::ExperimentConfiguration, args...; kwargs...) = get_model(econfig.model_config, args...; kwargs...)

function get_model(
    config::ImageClassificationModelConfiguration;
    seed::Int,
    device=gpu,
    warmup::Bool=true,  # Helps reduce Zygote compile times
)
    init_channel_size = config.num_channels[1]

    downsample_layers = [
        conv3x3(3 => init_channel_size; stride=config.downsample_times >= 1 ? 2 : 1),
        BatchNorm(init_channel_size, relu; affine=true, track_stats=false),
        conv3x3(init_channel_size => init_channel_size; stride=config.downsample_times >= 2 ? 2 : 1),
        BatchNorm(init_channel_size, relu; affine=true, track_stats=false),
    ]
    for _ in 3:(config.downsample_times)
        append!(
            downsample_layers,
            [
                conv3x3(init_channel_size => init_channel_size; stride=2),
                BatchNorm(init_channel_size, relu; affine=true, track_stats=false),
            ],
        )
    end
    downsample = Chain(downsample_layers...)

    stage0 = if config.downsample_times == 0 && config.num_branches <= 2
        NoOpLayer()
    else
        Chain(
            conv1x1(init_channel_size => init_channel_size; bias=false),
            BatchNorm(init_channel_size, relu; affine=true, track_stats=false),
        )
    end

    initial_layers = Chain(downsample, stage0)

    main_layers = Tuple(
        ResidualBlockV1(
            config.num_channels[i] => config.num_channels[i];
            deq_expand=config.expansion_factor,
            dropout_rate=config.dropout_rate,
            num_gn_groups=config.group_count,
            n_big_kernels=config.big_kernels[i],
        ) for i in 1:(config.num_branches)
    )

    mapping_layers = Matrix{Lux.AbstractExplicitLayer}(undef, config.num_branches, config.num_branches)
    for i in 1:(config.num_branches)
        for j in 1:(config.num_branches)
            if i == j
                mapping_layers[i, j] = NoOpLayer()
            elseif i < j
                mapping_layers[i, j] = downsample_module(
                    config.num_channels[i] => config.num_channels[j], j - i, relu; group_count=config.group_count
                )
            else
                mapping_layers[i, j] = upsample_module(
                    config.num_channels[i] => config.num_channels[j],
                    i - j,
                    relu;
                    group_count=config.group_count,
                    upsample_mode=:nearest,
                )
            end
        end
    end

    post_fuse_layers = Tuple(
        Chain(
            ActivationFunction(relu),
            conv1x1(config.num_channels[i] => config.num_channels[i]),
            # GroupNorm(config.num_channels[i], config.group_count ÷ 2; affine=true, track_stats=false),
            BatchNorm(config.num_channels[i]; affine=true, track_stats=false),
        ) for i in 1:(config.num_branches)
    )

    increment_modules = Parallel(
        nothing,
        [BottleneckBlockV2(config.num_channels[i] => config.head_channels[i]) for i in 1:(config.num_branches)]...,
    )

    downsample_modules = PairwiseFusion(
        config.fuse_method == :sum ? (+) : error("Only `fuse_method` = `:sum` is supported"),
        [
            Chain(
                conv3x3(config.head_channels[i] * 4 => config.head_channels[i + 1] * 4; stride=2, bias=true),
                BatchNorm(config.head_channels[i + 1] * 4, relu; track_stats=false, affine=true),
            ) for i in 1:(config.num_branches - 1)
        ]...,
    )

    final_layers = Chain(
        increment_modules,
        downsample_modules,
        conv1x1(config.head_channels[config.num_branches] * 4 => config.final_channelsize; bias=true),
        BatchNorm(config.final_channelsize, relu; track_stats=false, affine=true),
        GlobalMeanPool(),
        FlattenLayer(),
        Dense(config.final_channelsize, config.num_classes),
    )

    solver = if config.continuous
        ContinuousDEQSolver(
            config.ode_solver;
            mode=config.stop_mode,
            abstol=1.0f-5,
            reltol=1.0f-5,
            abstol_termination=config.abstol,
            reltol_termination=config.reltol,
        )
    else
        error("Discrete Solvers have not been updated yet")
    end

    sensealg = SteadyStateAdjoint(config.abstol, config.reltol, config.bwd_maxiters)

    deq = if config.model_type ∈ (:SKIP, :SKIPV2)
        shortcut = if config.model_type == :SKIP
            slayers = Lux.AbstractExplicitLayer[ResidualBlockV2(config.num_channels[1] => config.num_channels[1])]
            for i in 1:(config.num_branches - 1)
                push!(
                    slayers,
                    downsample_module(
                        config.num_channels[1] => config.num_channels[i + 1],
                        i,
                        relu;
                        group_count=config.group_count,
                    ),
                )
            end
            tuple(slayers...)
        else
            nothing
        end
        MultiScaleSkipDeepEquilibriumNetwork(
            main_layers,
            mapping_layers,
            post_fuse_layers,
            shortcut,
            solver,
            compute_feature_scales(config);
            maxiters=config.fwd_maxiters,
            sensealg=sensealg,
            verbose=false,
        )
    elseif config.model_type == :VANILLA
        MultiScaleDeepEquilibriumNetwork(
            main_layers,
            mapping_layers,
            post_fuse_layers,
            solver,
            compute_feature_scales(config);
            maxiters=config.fwd_maxiters,
            sensealg=sensealg,
            verbose=false,
        )
    else
        throw(ArgumentError("`model_type` must be one of `[:SKIP, :SKIPV2, :VANILLA]`"))
    end

    model = DEQChain(initial_layers, deq, final_layers)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    ps, st = device.(Lux.setup(rng, model))

    if warmup
        clean_println("Starting Model Warmup")
        x__ = device(randn(Float32, config.image_size..., 3, 2))
        y__ = device(Float32.(onehotbatch([1, 2], 0:(config.num_classes - 1))))
        model(x__, ps, st)
        clean_println("Forward Pass Warmup Completed")

        st_ = Lux.update_state(st, :fixed_depth, 2)
        model(x__, ps, st_)
        clean_println("Forward Pass (Pretraining) Warmup Completed")

        lfn = loss_function(config)
        (l, _, _, _), back = pullback(p -> lfn(x__, y__, model, p, st), ps)
        back((one(l), nothing, nothing, nothing))
        clean_println("Backward Pass Warmup Completed")

        (l, _, _, _), back = pullback(p -> lfn(x__, y__, model, p, st_), ps)
        back((one(l), nothing, nothing, nothing))
        clean_println("Backward Pass (Pretraining) Warmup Completed")

        invoke_gc()
    end

    ps, st = if is_distributed()
        ps_ = FluxMPI.synchronize!(ps; root_rank=0)
        st_ = FluxMPI.synchronize!(st; root_rank=0)
        ps_, st_
    else
        ps, st
    end

    return model, ps, st
end
