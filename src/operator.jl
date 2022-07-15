import LinearAlgebra
import SciMLBase

"""
    ZygotePullbackMultiplyOperator{T, typeof(pullback), typeof(shape)}(pullback, shape)

Constructs an operator for computing the VJP when operated (via `mul!` or `*`) on a vector.
"""
struct ZygotePullbackMultiplyOperator{T, F, S}
  f::F
  s::S
end

Base.deepcopy(op::ZygotePullbackMultiplyOperator) = op

Base.size(z::ZygotePullbackMultiplyOperator) = (prod(z.s), prod(z.s))
Base.size(z::ZygotePullbackMultiplyOperator, ::Int64) = prod(z.s)

Base.eltype(::ZygotePullbackMultiplyOperator{T}) where {T} = T

function LinearAlgebra.mul!(du::AbstractVector, L::ZygotePullbackMultiplyOperator,
                            x::AbstractVector)
  return du .= vec(L * x)
end

function Base.:*(L::ZygotePullbackMultiplyOperator, x::AbstractVector)
  return L.f(reshape(x, L.s))[1]
end

SciMLBase.isinplace(z::ZygotePullbackMultiplyOperator, ::Int64) = false
