# Functions
function conv3x3(mapping; stride::Int=1, bias::Bool=false)
    return Conv((3, 3), mapping; stride=stride, pad=1, bias=bias)
end

function conv5x5(mapping; stride::Int=1, bias::Bool=false)
    return Conv((5, 5), mapping; stride=stride, pad=2, bias=bias)
end

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
    conv1 = wn_layer((n_big_kernels >= 1 ? conv5x5 : conv3x3)(inplanes => inner_planes; stride=1, bias=false))
    gn1 = GroupNormV2(inner_planes, num_gn_groups, relu; affine=gn_affine, track_stats=false)

    conv2 = wn_layer((n_big_kernels >= 2 ? conv5x5 : conv3x3)(inner_planes => planes; stride=1, bias=false))
    gn2 = GroupNormV2(planes, num_gn_groups, relu; affine=gn_affine, track_stats=false)

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
    buf = Zygote.Buffer(typeof(x)[])
    push!(buf, bn.layers[1](x, injection))
    for (i, l) in enumerate(bn.layers[2:end])
        push!(buf, l(buf[i]))
    end
    return copy(buf)
end

function (bn::BranchNet)(x::AbstractArray{T}, injection::AbstractArray{T}, injections...) where {T}
    buf = Zygote.Buffer(typeof(x)[])
    push!(buf, bn.layers[1](x, injection))
    for (i, l) in enumerate(bn.layers[2:end])
        push!(buf, l(buf[i], injections[i]))
    end
    return copy(buf)
end

function (bn::BranchNet)(x::AbstractArray{T}, injection::T=T(0)) where {T}
    buf = Zygote.Buffer(typeof(x)[])
    push!(buf, bn.layers[1](x))
    for (i, l) in enumerate(bn.layers[2:end])
        push!(buf, l(buf[i]))
    end
    return copy(buf)
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
    buf = Zygote.Buffer([])
    for l in mpn.layers
        push!(buf, l(x...))
    end
    return copy(buf)
end

function (mpn::MultiParallelNet)(args...)
    buf = Zygote.Buffer([])
    for l in mpn.layers
        push!(buf, l(args...))
    end
    return copy(buf)
end

# Downsample Module
function downsample_module(in_channels::Int, out_channels::Int, in_resolution::Int, out_resolution::Int;
                           num_groups::Int=4, gn_affine::Bool=true)
    @assert in_resolution > out_resolution
    level_diff = Int(log2(in_resolution ÷ out_resolution))
    @assert ispow2(level_diff)

    layers = []

    for i in 1:level_diff
        intermediate_channels = i == level_diff ? out_channels : in_channels
        push!(layers, Conv((3, 3), in_channels => intermediate_channels; stride=2, pad=1, bias=false))
        push!(layers,
              GroupNormV2(intermediate_channels, num_groups, i == level_diff ? identity : relu; affine=gn_affine,
                          track_stats=false))
    end

    return Chain(layers...)
end

# Upsample Module
function upsample_module(in_channels::Int, out_channels::Int, in_resolution::Int, out_resolution::Int;
                         num_groups::Int=4, gn_affine::Bool=true)
    @assert in_resolution < out_resolution
    level_diff = Int(log2(out_resolution ÷ in_resolution))
    @assert ispow2(level_diff)

    return Chain(Conv((1, 1), in_channels => out_channels; bias=false),
                 GroupNormV2(out_channels, num_groups, relu; affine=gn_affine, track_stats=false),
                 Upsample(:nearest; scale=2^level_diff))
end

# Mapping Module
function expand_channels_module(in_channels::Int, out_channels::Int; num_groups::Int=4, gn_affine::Bool=true)
    return Chain(conv3x3(in_channels => out_channels; bias=false),
                 GroupNormV2(out_channels, num_groups, relu; affine=gn_affine, track_stats=false))
end
