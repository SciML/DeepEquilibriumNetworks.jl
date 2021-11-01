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

function SciMLBase.recursivecopy(a::MultiResolutionFeatures)
    return construct(
        MultiResolutionFeatures,
        map(x -> SingleResolutionFeatures(copy(x.values)), a.nodes),
    )
end

DiffEqBase.UNITLESS_ABS2(a::MultiResolutionFeatures) =
    sum(x -> sum(abs2, x.values), a.nodes)::eltype(a)

Base.zero(s::SingleResolutionFeatures) =
    SingleResolutionFeatures(zero(s.values))

Base.zero(v::MultiResolutionFeatures) =
    construct(MultiResolutionFeatures, zero.(v.nodes))

DiffEqBase.NAN_CHECK(v::MultiResolutionFeatures) =
    any(x -> DiffEqBase.NAN_CHECK(x.values), v.nodes)

Base.any(v::MultiResolutionFeatures) =
    any(x -> any(x.values), v.nodes)

Base.all(v::MultiResolutionFeatures) =
    all(x -> all(x.values), v.nodes)