# For MultiScale DEQs
"""
    split_and_reshape(x::AbstractMatrix, ::Sidxs, ::Sshapes)

Splits up the AbstractMatrix into chunks and reshapes them.

## Arguments

  - `x`: Matrix to be split up.
  - `Sidxs`: Indices to partition the array at. (must be a `static` type).
  - `Sshapes`: Shapes to reshape the split the arrays. (must be a `static` type).

## Example

```@example
using DeepEquilibriumNetworks, Static

x1 = ones(Float32, 4, 4)
x2 = fill!(zeros(Float32, 2, 4), 0.5f0)
x3 = zeros(Float32, 1, 4)

x = vcat(x1, x2, x3)
split_idxs = static(cumsum((0, size(x1, 1), size(x2, 1), size(x3, 1))))
shapes = static((size(x1, 1), size(x2, 1), size(x3, 1)))

DEQs.split_and_reshape(x, split_idxs, shapes)
```
"""
@generated function split_and_reshape(x::AbstractMatrix, ::T, ::S) where {T, S}
  idxs, shapes = known(T), known(S)
  dims = [reshape((idxs[i] + 1):idxs[i + 1], shapes[i]...) for i in 1:(length(idxs) - 1)]
  varnames = [gensym("x_view") for _ in dims]
  calls = []
  for (i, dim) in enumerate(dims)
    push!(calls, :($(varnames[i]) = view(x, $dim, :)))
  end
  push!(calls, :(return tuple($(varnames...))))
  return Expr(:block, calls...)
end

# General Utils
"""
    init_identity_matrix(x::AbstractArray, scale::T=1)

Create an identity matrix of shape `[length(x), length(x)]` and placed on the same device
as `x`, and scale the matrix by `scale`.
"""
@inline function init_identity_matrix(x::AbstractArray{T}, scale::T=T(1)) where {T}
  x_ = vec(x)
  return _init_identity_matrix!(x_ .* x_', scale)
end

@inline function _init_identity_matrix!(x::AbstractMatrix{T}, scale::T=T(1)) where {T}
  x .= zero(T)
  view(x, LinearAlgebra.diagind(x)) .= scale .* true
  return x
end
