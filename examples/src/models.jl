# Building Blocks
## Helpful Functional Wrappers
function conv1x1(mapping, activation=identity; stride::Int=1, bias=false, kwargs...)
    return EFL.Conv((1, 1), mapping, activation; stride=stride, pad=0, bias=bias, kwargs...)
end

function conv3x3(mapping, activation=identity; stride::Int=1, bias=false, kwargs...)
    return EFL.Conv((3, 3), mapping, activation; stride=stride, pad=1, bias=bias, kwargs...)
end

function conv5x5(mapping, activation=identity; stride::Int=1, bias=false, kwargs...)
    return EFL.Conv((5, 5), mapping, activation; stride=stride, pad=2, bias=bias, kwargs...)
end

reassociate(x::NTuple{2,<:AbstractArray}, y) = (x[1], (x[2], y))

## Downsample Module
function downsample_module(mapping, resolution_mapping, activation; group_count=8)
    in_resolution, out_resolution = resolution_mapping
    in_channels, out_channels = mapping
    @assert in_resolution > out_resolution
    @assert ispow2(in_resolution ÷ out_resolution)
    level_diff = Int(log2(in_resolution ÷ out_resolution))

    function intermediate_mapping(i)
        if in_channels * (2^level_diff) == out_channels
            return (in_channels * (2^(i - 1))) => (in_channels * (2^i))
        else
            return i == level_diff ? in_channels => out_channels : in_channels => in_channels
        end
    end

    layers = EFL.AbstractExplicitLayer[]
    for i in 1:level_diff
        inchs, outchs = intermediate_mapping(i)
        push!(layers, conv3x3(inchs => outchs; stride=2, initW=NormalInitializer()))
        push!(layers, EFL.GroupNorm(outchs, group_count, activation; affine=true, track_stats=false))
    end
    return EFL.Chain(layers...)
end

## Upsample Module
function upsample_module(mapping, resolution_mapping, activation; upsample_mode::Symbol=:nearest, group_count=8)
    in_resolution, out_resolution = resolution_mapping
    in_channels, out_channels = mapping
    @assert in_resolution < out_resolution
    @assert ispow2(out_resolution ÷ in_resolution)
    level_diff = Int(log2(out_resolution ÷ in_resolution))

    function intermediate_mapping(i)
        if out_channels * (2^level_diff) == in_channels
            (in_channels ÷ (2^(i - 1))) => (in_channels ÷ (2^i))
        else
            i == level_diff ? in_channels => out_channels : in_channels => in_channels
        end
    end

    layers = EFL.AbstractExplicitLayer[]
    for i in 1:level_diff
        inchs, outchs = intermediate_mapping(i)
        push!(layers, conv1x1(inchs => outchs; initW=NormalInitializer()))
        push!(layers, EFL.GroupNorm(outchs, group_count, activation; affine=true, track_stats=false))
        push!(layers, EFL.Upsample(upsample_mode; scale=2))
    end
    return EFL.Chain(layers...)
end

## Residual Block
function ResidualBlockV1(
    mapping;
    deq_expand::Int=5,
    num_gn_groups::Int=4,
    downsample=EFL.NoOpLayer(),
    n_big_kernels::Int=0,
    dropout_rate::Real=0.0f0,
    gn_affine::Bool=true,
    weight_norm::Bool=true,
    gn_track_stats::Bool=false,
)
    inplanes, outplanes = mapping
    inner_planes = outplanes * deq_expand
    conv1 = (n_big_kernels >= 1 ? conv5x5 : conv3x3)(inplanes => inner_planes; initW=NormalInitializer(), bias=false)
    conv2 = (n_big_kernels >= 2 ? conv5x5 : conv3x3)(inner_planes => outplanes; initW=NormalInitializer(), bias=false)

    conv1, conv2 = if weight_norm
        EFL.WeightNorm(conv1, (:weight,)), EFL.WeightNorm(conv2, (:weight,))
    else
        conv1, conv2
    end

    gn1 = EFL.GroupNorm(inner_planes, num_gn_groups, gelu; affine=gn_affine, track_stats=gn_track_stats)
    gn2 = EFL.GroupNorm(outplanes, num_gn_groups, gelu; affine=gn_affine, track_stats=gn_track_stats)
    gn3 = EFL.GroupNorm(outplanes, num_gn_groups; affine=gn_affine, track_stats=gn_track_stats)

    dropout = iszero(dropout_rate) ? EFL.NoOpLayer() : EFL.Dropout(dropout_rate)

    return EFL.Chain(
        EFL.Parallel(
            reassociate, # Reassociate and Merge
            EFL.Chain(conv1, gn1, conv2, EFL.BranchLayer(downsample, dropout)),  # For x
            EFL.NoOpLayer(),  # For injection
        ),
        EFL.Parallel(
            +,
            EFL.NoOpLayer(),  # For y1
            EFL.Chain(
                EFL.WrappedFunction(y2i -> y2i[1] .+ y2i[2]),  # Since injection could be a scalar
                gn2,
            ),  # For (y2, injection)
        ),
        EFL.WrappedFunction(Base.Fix1(broadcast, gelu)),
        gn3,
    )
end

function ResidualBlockV2(
    mapping;
    deq_expand::Int=5,
    num_gn_groups::Int=4,
    downsample=EFL.NoOpLayer(),
    n_big_kernels::Int=0,
    dropout_rate::Real=0.0f0,
    gn_affine::Bool=true,
    weight_norm::Bool=true,
    gn_track_stats::Bool=false,
)
    inplanes, outplanes = mapping
    inner_planes = outplanes * deq_expand
    conv1 = (n_big_kernels >= 1 ? conv5x5 : conv3x3)(inplanes => inner_planes; initW=NormalInitializer(), bias=false)
    conv2 = (n_big_kernels >= 2 ? conv5x5 : conv3x3)(inner_planes => outplanes; initW=NormalInitializer(), bias=false)

    conv1, conv2 = if weight_norm
        EFL.WeightNorm(conv1, (:weight,)), EFL.WeightNorm(conv2, (:weight,))
    else
        conv1, conv2
    end

    gn1 = EFL.GroupNorm(inner_planes, num_gn_groups, gelu; affine=gn_affine, track_stats=gn_track_stats)
    gn2 = EFL.GroupNorm(outplanes, num_gn_groups, gelu; affine=gn_affine, track_stats=gn_track_stats)
    gn3 = EFL.GroupNorm(outplanes, num_gn_groups; affine=gn_affine, track_stats=gn_track_stats)

    dropout = iszero(dropout_rate) ? EFL.NoOpLayer() : EFL.Dropout(dropout_rate)

    return EFL.Chain(
        conv1,
        gn1,
        conv2,
        EFL.BranchLayer(downsample, dropout),
        EFL.Parallel(+, EFL.NoOpLayer(), gn2),
        EFL.WrappedFunction(Base.Fix1(broadcast, gelu)),
        gn3,
    )
end

function BottleneckBlockV1(mapping::Pair, expansion::Int=4; bn_track_stats::Bool=true, bn_affine::Bool=true)
    rescale = if first(mapping) != last(mapping) * expansion
        EFL.Chain(
            conv1x1(first(mapping) => last(mapping) * expansion; initW=NormalInitializer()),
            EFL.BatchNorm(last(mapping) * expansion; track_stats=bn_track_stats, affine=bn_affine),
        )
    else
        EFL.NoOpLayer()
    end

    return EFL.Chain(
        EFL.Parallel(
            reassociate, EFL.BranchLayer(rescale, conv1x1(mapping; initW=NormalInitializer())), EFL.NoOpLayer()
        ),
        EFL.Parallel(
            +,
            EFL.NoOpLayer(),
            EFL.Chain(
                EFL.WrappedFunction(y2i -> y2i[1] .+ y2i[2]),  # Since injection could be a scalar
                EFL.Chain(
                    EFL.BatchNorm(last(mapping), gelu; affine=bn_affine, track_stats=bn_track_stats),
                    conv3x3(last(mapping) => last(mapping) * expansion; initW=NormalInitializer()),
                    EFL.BatchNorm(last(mapping) * expansion, gelu; track_stats=bn_track_stats, affine=bn_affine),
                    conv1x1(last(mapping) * expansion => last(mapping) * expansion; initW=NormalInitializer()),
                    EFL.BatchNorm(last(mapping) * expansion; track_stats=bn_track_stats, affine=bn_affine),
                ),
            ),
        ),
        EFL.WrappedFunction(Base.Fix1(broadcast, gelu)),
    )
end

function BottleneckBlockV2(mapping::Pair, expansion::Int=4; bn_track_stats::Bool=true, bn_affine::Bool=true)
    rescale = if first(mapping) != last(mapping) * expansion
        EFL.Chain(
            conv1x1(first(mapping) => last(mapping) * expansion; initW=NormalInitializer()),
            EFL.BatchNorm(last(mapping) * expansion; track_stats=bn_track_stats, affine=bn_affine),
        )
    else
        EFL.NoOpLayer()
    end

    return EFL.Chain(
        EFL.Parallel(
            +,
            rescale,
            EFL.Chain(
                conv1x1(mapping; initW=NormalInitializer()),
                EFL.BatchNorm(last(mapping), gelu; affine=bn_affine, track_stats=bn_track_stats),
                conv3x3(last(mapping) => last(mapping) * expansion; initW=NormalInitializer()),
                EFL.BatchNorm(last(mapping) * expansion, gelu; track_stats=bn_track_stats, affine=bn_affine),
                conv1x1(last(mapping) * expansion => last(mapping) * expansion; initW=NormalInitializer()),
                EFL.BatchNorm(last(mapping) * expansion; track_stats=bn_track_stats, affine=bn_affine),
            ),
        ),
        EFL.WrappedFunction(Base.Fix1(broadcast, gelu)),
    )
end

# Dataset Specific Models
## CIFAR10 -- MultiScaleDEQ
function get_model(
    ::Val{:CIFAR10};
    dropout_rate,
    group_count=8,
    model_type::Symbol,
    continuous::Bool=true,
    maxiters::Int,
    abstol,
    reltol,
    seed,
    device=gpu,
    warmup::Bool=true,  # Helps reduce time for Zygote to compile gradients first time
)
    initial_layers = EFL.Chain(
        conv3x3(3 => 24; initW=NormalInitializer()),
        EFL.BatchNorm(24, gelu; track_stats=true, affine=true),
        conv3x3(24 => 24; initW=NormalInitializer()),
        EFL.BatchNorm(24, gelu; track_stats=true, affine=true),
    )

    main_layers = (
        ResidualBlockV1(24 => 24; dropout_rate, num_gn_groups=group_count),  # 32 x 32
        ResidualBlockV1(24 => 24; dropout_rate, num_gn_groups=group_count),  # 16 x 16
    )

    mapping_layers = [
        EFL.NoOpLayer() downsample_module(24 => 24, 32 => 16, gelu; group_count=group_count)
        upsample_module(24 => 24, 16 => 32, gelu; group_count=group_count, upsample_mode=:nearest) EFL.NoOpLayer()
    ]

    post_fuse_layers = (
        EFL.Chain(
            EFL.WrappedFunction(Base.Fix1(broadcast, gelu)),
            conv1x1(24 => 24; initW=NormalInitializer()),
            EFL.GroupNorm(24, group_count ÷ 2; affine=true, track_stats=false),
        ),
        EFL.Chain(
            EFL.WrappedFunction(Base.Fix1(broadcast, gelu)),
            conv1x1(24 => 24; initW=NormalInitializer()),
            EFL.GroupNorm(24, group_count ÷ 2; affine=true, track_stats=false),
        ),
    )

    final_layers = EFL.Chain(
        EFL.Parallel(
            +,
            EFL.Chain(
                BottleneckBlockV2(24 => 8),
                conv3x3(8 * 4 => 16 * 4; stride=2, initW=NormalInitializer()),
                EFL.BatchNorm(16 * 4, gelu; track_stats=true, affine=true),
            ),
            BottleneckBlockV2(24 => 16, 4),
        ),
        conv1x1(16 * 4 => 200; initW=NormalInitializer()),
        EFL.BatchNorm(200, gelu; track_stats=true, affine=true),
        EFL.GlobalMeanPool(),
        EFL.FlattenLayer(),
        EFL.Dense(200, 10),
    )

    solver = if continuous
        ContinuousDEQSolver(
            VCABM3();
            mode=:rel_deq_best,
            abstol=abstol,
            reltol=reltol,
            abstol_termination=abstol,
            reltol_termination=reltol,
        )
    else
        error("Discrete Solvers have not been updated yet")
    end

    sensealg = SteadyStateAdjoint(abstol, reltol, min(maxiters, 15))

    deq = if model_type ∈ (:skip, :skipv2)
        shortcut = if model_type == :skip
            (
            ResidualBlockV2(24 => 24; num_gn_groups=group_count),
            downsample_module(24 => 24, 32 => 16, gelu; group_count=group_count),
        )
        else
            nothing
        end
        MultiScaleSkipDeepEquilibriumNetwork(
            main_layers,
            mapping_layers,
            post_fuse_layers,
            shortcut,
            solver,
            ((32, 32, 24), (16, 16, 24));
            maxiters=maxiters,
            sensealg=sensealg,
            verbose=false,
        )
    elseif model_type == :vanilla
        MultiScaleSkipDeepEquilibriumNetwork(
            main_layers,
            mapping_layers,
            post_fuse_layers,
            solver,
            ((32, 32, 24), (16, 16, 24));
            maxiters=maxiters,
            sensealg=sensealg,
            verbose=false,
        )
    else
        throw(ArgumentError("`model_type` must be one of `[:skip, :skipv2, :vanilla]`"))
    end

    model = DEQChain(initial_layers, deq, final_layers)
    ps, st = EFL.setup(MersenneTwister(seed), model) .|> device

    if warmup
        clean_println("Starting Model Warmup")
        x__ = randn(Float32, 32, 32, 3, 1) |> device
        y__ = Float32.(Flux.onehotbatch([1], 0:9)) |> device
        model(x__, ps, st)
        clean_println("Forward Pass Warmup Completed")
        lfn = loss_function(:CIFAR10, model_type)
        (l, _, _), back = Flux.pullback(p -> lfn(x__, y__, model, p, st), ps)
        back((one(l), nothing, nothing))
        clean_println("Backward Pass Warmup Completed")
    end

    return model, ps, st
end
