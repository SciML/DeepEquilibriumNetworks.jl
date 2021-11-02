# Code adapted from https://github.com/Chemellia/AtomicGraphNets.jl/blob/main/src/layers.jl
# Layers have been GPUified (will try to get them upstreamed)
# AGN stands for AtomicGraphNets
abstract type AtomicGraphLayer end

(l::AtomicGraphLayer)(x::Tuple) = l(x[1], x[2])
(l::AtomicGraphLayer)(fa::FeaturizedAtoms) =
    l(fa.atoms.laplacian, fa.encoded_features)

struct AGNConv{W,B,F} <: AtomicGraphLayer
    selfweight::W
    convweight::W
    bias::B
    σ::F
end

function AGNConv(
    ch::Pair{<:Integer,<:Integer},
    σ = softplus;
    initW = Flux.glorot_uniform,
    initb = zeros,
    T::DataType = Float32,
)
    selfweight = T.(initW(ch[2], ch[1]))
    convweight = T.(initW(ch[2], ch[1]))
    b = T.(initb(ch[2], 1))
    return AGNConv(selfweight, convweight, b, σ)
end

@functor AGNConv

function (l::AGNConv)(lapl::AbstractMatrix, X::AbstractMatrix)
    out_mat = Flux.normalise(
        l.σ.(l.convweight * X * lapl .+ l.selfweight * X .+ l.bias),
        dims = [1, 2],
    )
    return lapl, out_mat
end

struct AGNMaxPool <: AtomicGraphLayer
    dim::Int64
    stride::Int64
    pad::Int64
end

struct AGNMeanPool <: AtomicGraphLayer
    dim::Int64
    stride::Int64
    pad::Int64
end

function AGNPool(
    pool_type::Symbol,
    in_num_features::Int64,
    out_num_features::Int64,
    pool_width_frac::Float64,
)
    dim, stride, pad = compute_pool_params(
        in_num_features,
        out_num_features,
        Float64(pool_width_frac),
    )
    if pool_type == :max
        T = AGNMaxPool
    elseif pool_type == :mean
        T = AGNMeanPool
    end
    return T(dim, stride, pad)
end

AGNMaxPool(args...) = AGNPool(:max, args...)

AGNMeanPool(args...) = AGNPool(:mean, args...)

function (m::AGNMaxPool)(lapl::AbstractMatrix, X::AbstractMatrix)
    x = reshape(X, (size(X)..., 1, 1))
    pdims =
        PoolDims(x, (m.dim, 1); padding = (m.pad, 0), stride = (m.stride, 1))
    return mean(maxpool(x, pdims), dims = 2)[:, :, 1, 1]
end

function (m::AGNMaxPool)(lapl::AbstractMatrix, X::AbstractMatrix)
    x = reshape(X, (size(X)..., 1, 1))
    pdims =
        PoolDims(x, (m.dim, 1); padding = (m.pad, 0), stride = (m.stride, 1))
    return mean(meanpool(x, pdims), dims = 2)[:, :, 1, 1]
end

# Batching Utilities
"""
    batch_graph_data(laplacians, encoded_features)

Takes vectors of laplacians and encoded features and joins them
into a single graph of disjoint subgraphs. The resulting graph
is massive and hence the return types are sparse. Few of the layers
don't work with Sparse Arrays (specifically on GPUs), so it would
make sense to convert them to dense.
"""
function batch_graph_data(laplacians, encoded_features)
    _sizes = map(x -> size(x, 1), laplacians)
    total_nodes = sum(_sizes)
    batched_laplacian =
        sparse(zeros(eltype(laplacians[1]), total_nodes, total_nodes))
    idx = 1
    for i = 1:length(laplacians)
        batched_laplacian[idx:idx + _sizes[i] - 1, idx:idx + _sizes[i] - 1] .= laplacians[i]
        idx += _sizes[i]
    end
    return batched_laplacian, sparse(hcat(encoded_features...))
end
