# For MultiScale DEQs
"""
    split_and_reshape(x::AbstractMatrix, ::Val{idxs}, ::Val{shapes}) where {idxs, shapes}

Splits up the AbstractMatrix into chunks and reshapes them.

## Arguments

  - `x`: Matrix to be split up.
  - `Sidxs`: Indices to partition the array at. (must be a `Val` type).
  - `Sshapes`: Shapes to reshape the split the arrays. (must be a `Val` type).

## Example

```@example
using DeepEquilibriumNetworks, Static

x1 = ones(Float32, 4, 4)
x2 = fill!(zeros(Float32, 2, 4), 0.5f0)
x3 = zeros(Float32, 1, 4)

x = vcat(x1, x2, x3)
split_idxs = Val(cumsum((0, size(x1, 1), size(x2, 1), size(x3, 1))))
shapes = Val((size(x1, 1), size(x2, 1), size(x3, 1)))

DEQs.split_and_reshape(x, split_idxs, shapes)
```
"""
@generated function split_and_reshape(x::AbstractMatrix, ::Val{idxs},
        ::Val{shapes}) where {idxs, shapes}
    dims = [reshape((idxs[i] + 1):idxs[i + 1], shapes[i]...) for i in 1:(length(idxs) - 1)]
    varnames = [gensym("x_view") for _ in dims]
    calls = []
    for (i, dim) in enumerate(dims)
        push!(calls, :($(varnames[i]) = view(x, $dim, :)))
    end
    push!(calls, :(return tuple($(varnames...))))
    return Expr(:block, calls...)
end

@inline flatten(x::AbstractVector) = reshape(x, length(x), 1)
@inline flatten(x::AbstractMatrix) = x
@inline flatten(x::AbstractArray) = reshape(x, :, size(x, ndims(x)))
