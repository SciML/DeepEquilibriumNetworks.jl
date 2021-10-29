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


# See https://github.com/FluxML/Flux.jl/issues/1733
# Not a very classy solution but it works
function parameter_destructure(m)
    xs = Zygote.Buffer([])
    ks = keys(IdDict(p => nothing for p in Flux.params(m)))
    fmap(m) do x
        x isa AbstractArray && x ∈ ks && push!(xs, copy(vec(x)))
        return x
    end
    return vcat(xs...), p -> _parameter_restructure(m, p)
end

function _parameter_restructure(m, xs)
    i = 0
    ks = keys(IdDict(p => nothing for p in Flux.params(m)))
    m̄ = fmap(m) do x
        (x isa AbstractArray && x ∈ ks) || return x
        x = reshape(xs[i.+(1:length(x))], size(x))
        i += length(x)
        return x
    end
    return m̄
end

Zygote.@adjoint function _parameter_restructure(m, xs)
    m̄, numel = _parameter_restructure(m, xs), length(xs)
    function _parameter_restructure_pullback(dm)
        xs′ = parameter_destructure(dm)[1]
        return (nothing, xs′)
    end
    return m̄, _parameter_restructure_pullback
end


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
