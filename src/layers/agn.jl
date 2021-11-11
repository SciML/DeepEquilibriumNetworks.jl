# Code adapted from https://github.com/Chemellia/AtomicGraphNets.jl/blob/main/src/layers.jl
# Layers have been GPUified (will try to get them upstreamed)
# AGN stands for AtomicGraphNets
pool_out_features(num_f::Int64, dim::Int64, stride::Int64, pad::Int64) =
    Int64(floor((num_f + 2 * pad - dim) / stride + 1))

function compute_pool_params(
    num_f_in::Int64,
    num_f_out::Int64,
    dim_frac::AbstractFloat;
    start_dim = Int64(round(dim_frac * num_f_in)),
    start_str = Int64(floor(num_f_in / num_f_out)),
)
    # take starting guesses
    dim = start_dim
    str = start_str
    p_numer = str * (num_f_out - 1) - (num_f_in - dim)
    if p_numer < 0
        p_numer == -1 ? dim = dim + 1 : str = str + 1
    end
    p_numer = str * (num_f_out - 1) - (num_f_in - dim)
    if p_numer < 0
        error("problem, negative p!")
    end
    if p_numer % 2 == 0
        pad = Int64(p_numer / 2)
    else
        dim = dim - 1
        pad = Int64((str * (num_f_out - 1) - (num_f_in - dim)) / 2)
    end
    out_fea_len = pool_out_features(num_f_in, dim, str, pad)
    if !(out_fea_len == num_f_out)
        print("problem, output feature wrong length!")
    end
    # check if pad gets comparable to width...
    if pad >= 0.8 * dim
        @warn "specified pooling width was hard to satisfy without nonsensically large padding relative to width, had to increase from desired width"
        dim, str, pad = compute_pool_params(
            num_f_in,
            num_f_out,
            dim_frac,
            start_dim = Int64(round(1.2 * start_dim)),
        )
    end
    dim, str, pad
end

abstract type AtomicGraphLayer end

(l::AtomicGraphLayer)(x::Tuple) = l(x[1], x[2])

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

function (m::AGNMeanPool)(lapl::AbstractMatrix, X::AbstractMatrix)
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
struct BatchedAtomicGraph{T1,T2,S}
    laplacians::T1
    encoded_features::T2
    sizes::S
end

Flux.@functor BatchedAtomicGraph (laplacians, encoded_features)

function batch_graph_data(laplacians, encoded_features)
    _sizes = map(x -> size(x, 1), laplacians)
    total_nodes = sum(_sizes)
    batched_laplacian =
        sparse(zeros(eltype(laplacians[1]), total_nodes, total_nodes))
    idx = 1
    for i = 1:length(laplacians)
        batched_laplacian[idx:idx+_sizes[i]-1, idx:idx+_sizes[i]-1] .=
            laplacians[i]
        idx += _sizes[i]
    end
    return BatchedAtomicGraph(
        batched_laplacian,
        sparse(hcat(encoded_features...)),
        vcat(0, cumsum(_sizes)),
    )
end
