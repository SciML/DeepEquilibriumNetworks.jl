# Basic Residual Block
struct BasicResidualBlock{C1,C2,GN1,GN2,GN3,DO,DR}
    conv1::C1
    conv2::C2
    gn1::GN1
    gn2::GN2
    gn3::GN3
    downsample::DO
    dropout::DR
end

Flux.@functor BasicResidualBlock

function BasicResidualBlock(outdims::Tuple, inplanes::Int, planes::Int; deq_expand::Int=5, num_gn_groups::Int=4,
                            downsample=identity, n_big_kernels::Int=0, dropout_rate::Real=0.0f0, gn_affine::Bool=true,
                            weight_norm::Bool=true)
    wn_layer = weight_norm ? WeightNorm : identity

    inner_planes = planes * deq_expand
    conv1 = wn_layer((n_big_kernels >= 1 ? conv5x5 : conv3x3)(inplanes => inner_planes; stride=1, bias=false, init=normal_init()))
    gn1 = GroupNormV2(inner_planes, num_gn_groups, gelu; affine=gn_affine, track_stats=false)

    conv2 = wn_layer((n_big_kernels >= 2 ? conv5x5 : conv3x3)(inner_planes => planes; stride=1, bias=false, init=normal_init()))
    gn2 = GroupNormV2(planes, num_gn_groups, gelu; affine=gn_affine, track_stats=false)

    gn3 = GroupNormV2(planes, num_gn_groups; affine=gn_affine, track_stats=false)

    return BasicResidualBlock(conv1, conv2, gn1, gn2, gn3, downsample,
                              (iszero(dropout_rate) ? identity :
                               VariationalHiddenDropout(dropout_rate, (outdims..., planes, 1))))
end

(b::BasicResidualBlock)(t::Tuple) = b(t...)

function (b::BasicResidualBlock)(x::AbstractArray{T}, injection::Union{AbstractArray{V},T}=T(0)) where {T,V}
    x_ = x
    x = b.conv1(x)
    x = b.conv2(b.gn1(x))
    residual = b.downsample(x_)
    return b.gn3(relu.(b.gn2(b.dropout(x) .+ injection) .+ residual))
end

function Base.show(io::IO, l::BasicResidualBlock)
    p, _ = destructure_parameters(l)
    return print(io, string(typeof(l).name.name), "() ", string(length(p)), " Trainable Parameters")
end

# BranchNet
struct BranchNet{L}
    layers::L

    function BranchNet(args...)
        layers = tuple(args...)
        return new{typeof(layers)}(layers)
    end

    BranchNet(layers::Tuple) = new{typeof(layers)}(layers)
end

Flux.@functor BranchNet

function (bn::BranchNet)(x::AbstractArray{T}, injection::AbstractArray{T}) where {T}
    buf = Zygote.Buffer(Vector{typeof(x)}(undef, length(bn.layers)))
    buf[1] = bn.layers[1](x, injection)
    for (i, l) in enumerate(bn.layers[2:end])
        buf[i + 1] = l(buf[i])
    end
    return Tuple(copy(buf))
end

function (bn::BranchNet)(x::AbstractArray{T}, injection::AbstractArray{T}, injections...) where {T}
    buf = Zygote.Buffer(Vector{typeof(x)}(undef, length(bn.layers)))
    buf[1] = bn.layers[1](x, injection)
    for (i, l) in enumerate(bn.layers[2:end])
        buf[i + 1] = l(buf[i], injections[i])
    end
    return Tuple(copy(buf))
end

function (bn::BranchNet)(x::AbstractArray{T}, injection::T=T(0)) where {T}
    buf = Zygote.Buffer(Vector{typeof(x)}(undef, length(bn.layers)))
    buf[1] = bn.layers[1](x)
    for (i, l) in enumerate(bn.layers[2:end])
        buf[i + 1] = l(buf[i])
    end
    return Tuple(copy(buf))
end

# Multi Parallel Net
struct MultiParallelNet{L}
    layers::L

    function MultiParallelNet(args...)
        layers = tuple(args...)
        return new{typeof(layers)}(layers)
    end

    MultiParallelNet(layers::Tuple) = new{typeof(layers)}(layers)

    MultiParallelNet(layers::Vector) = MultiParallelNet(layers...)
end

Flux.@functor MultiParallelNet

function (mpn::MultiParallelNet)(x::Union{Tuple,Vector})
    buf = Zygote.Buffer(Vector{Any}(undef, length(mpn.layers)))
    for (i, l) in enumerate(mpn.layers)
        buf[i] = l(x...)
    end
    return Tuple(copy(buf))
end

function (mpn::MultiParallelNet)(args...)
    buf = Zygote.Buffer(Vector{Any}(undef, length(mpn.layers)))
    for (i, l) in enumerate(mpn.layers)
        buf[i] = l(args...)
    end
    return Tuple(copy(buf))
end

# Bottleneck Layer
struct BasicBottleneckBlock{C1,C2,D}
    conv1::C1
    chain::C2
    downsample::D
end

function Base.show(io::IO, l::BasicBottleneckBlock)
    p, _ = destructure_parameters(l)
    return print(io, string(typeof(l).name.name), "() ", string(length(p)), " Trainable Parameters")
end

Flux.@functor BasicBottleneckBlock

function BasicBottleneckBlock(mapping::Pair, expansion::Int=4)
    downsample = if first(mapping) != last(mapping) * expansion
        conv1x1_norm(first(mapping) => last(mapping) * expansion, identity; norm_layer=BatchNormV2,
                     norm_kwargs=Dict{Symbol,Any}(:track_stats => false, :affine => true),
                     conv_kwargs=Dict{Symbol,Any}(:bias => false))
    else
        identity
    end
    conv1 = conv1x1(mapping; bias=false, init=normal_init())
    chain = Chain(BatchNormV2(last(mapping), gelu; affine=true, track_stats=false),
                  conv3x3_norm(last(mapping) => last(mapping) * expansion, gelu; norm_layer=BatchNormV2,
                               conv_kwargs=Dict{Symbol,Any}(:bias => false, :init => normal_init()),
                               norm_kwargs=Dict{Symbol,Any}(:track_stats => false, :affine => true)).layers...,
                  conv1x1_norm(last(mapping) * expansion => last(mapping) * expansion; norm_layer=BatchNormV2,
                               conv_kwargs=Dict{Symbol,Any}(:bias => false, :init => normal_init()),
                               norm_kwargs=Dict{Symbol,Any}(:track_stats => false, :affine => true)).layers...)

    return BasicBottleneckBlock(conv1, chain, downsample)
end

function (bl::BasicBottleneckBlock)(x::AbstractArray{T}, injection::Union{AbstractArray{T},T}=T(0)) where {T}
    return gelu.(bl.chain(bl.conv1(x) .+ injection) .+ bl.downsample(x))
end
