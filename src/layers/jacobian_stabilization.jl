gaussian_like(p::Array) = randn(eltype(p), size(p))
gaussian_like(p::CuArray) = CUDA.randn(eltype(p), size(p))

Zygote.@nograd gaussian_like

"""
    compute_deq_jacobian_loss(re, p, z, x)

Computes Jacobian Stabilization Loss ([bai2021stabilizing](@cite)).

## Arguments

* `re`: Constructs the model given the parameters `p`.
* `p`: Parameters of the model.
* `z`: Steady State.
* `x`: Input to the model.

## Current Known Failure Modes

1. Conv layers error out due to ForwardDiff on GPUs
2. If the model internally uses destructure/restructure eg. `WeightNorm` Layer, then this loss function will error out in the backward pass.
"""
function compute_deq_jacobian_loss(re, p::AbstractVector{T}, z::A, x::A) where {T,A<:AbstractArray}
    d = length(z)
    v = gaussian_like(z)
    model = re(p)

    _, back = Zygote.pullback(model, z, x)
    vjp_z, vjp_x = back(v)
    # NOTE: This weird sum(zero, ...) ensures that we get zeros instead of nothings
    return sum(abs2, vjp_z) / d + sum(zero, vjp_x)
end
