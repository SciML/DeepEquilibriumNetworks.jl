# General Utils
function _init_identity_matrix(x::AbstractArray{T}, scale::T = T(1)) where {T}
    x_ = vec(x)
    return _init_identity_matrix!(x_ .* x_', scale)
end

function _init_identity_matrix!(x::AbstractMatrix{T}, scale::T = T(1)) where {T}
    x .= zero(T)
    idxs = diagind(x)
    @. @view(x[idxs]) = scale * true
    return x
end


# General DEQ Utils
Flux.trainable(deq::AbstractDeepEquilibriumNetwork) = (deq.p,)

function get_and_clear_nfe!(model::AbstractDeepEquilibriumNetwork)
    nfe = model.stats.nfe
    model.stats.nfe = 0
    return nfe
end

mutable struct DEQTrainingStats
    nfe::Int
end


# For MultiScale DEQs
struct SingleResolutionFeatures{A,B} <: AbstractMultiScaleArrayLeaf{B}
    values::A
end

function SingleResolutionFeatures(values::T) where {T}
    return SingleResolutionFeatures{T,eltype(values)}(values)
end

struct MultiResolutionFeatures{T<:AbstractMultiScaleArray,B<:Number} <:
       AbstractMultiScaleArrayHead{B}
    nodes::Vector{T}
    values::Vector{B}
    end_idxs::Vector{Int}
end

Base.eltype(::Type{MultiResolutionFeatures}) = Float32

Base.vec(v::MultiResolutionFeatures) = v[:]

Base.getindex(v::MultiResolutionFeatures, ::Colon) =
    vcat([x.values for x in v.nodes]...)

Base.similar(
    v::MultiResolutionFeatures{T1,T},
    dims::Union{Integer,AbstractUnitRange},
) where {T1,T<:Number} = similar(v.nodes[1].values, dims)

Base.similar(
    v::MultiResolutionFeatures{T1,T},
    dims::Tuple,
) where {T1,T<:Number} = similar(v.nodes[1].values, dims)
