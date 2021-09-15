## Ensures that the returned array is on the same device as `x`
init_filled_array(x, shape, value) = fill!(similar(x, shape), value)
init_zero(x, shape) = init_filled_array(x, shape, 0)

@views function init_identity_matrix(
    x::AbstractArray{T},
    n::Int,
    scale::T = T(1),
) where {T}
    x_ = similar(x, n)
    Id = x_ .* x_' .* false
    idxs = diagind(Id)
    @. Id[idxs] = scale * true
    return Id
end