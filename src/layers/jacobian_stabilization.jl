gaussian_like(p::Array) = randn(eltype(p), size(p))
gaussian_like(p::CuArray) = CUDA.randn(eltype(p), size(p))

Zygote.@nograd gaussian_like

# NOTE: If the model internally uses destructure/restructure eg. WeightNorm Layer, then
#       this loss function will error out in the backward pass.
# FIXME: Conv layers error out due to ForwardDiff on GPUs
function compute_deq_jacobian_loss(re, p::AbstractVector{T}, z::A, x::A) where {T,A<:AbstractArray}
    d = length(z)
    v = gaussian_like(z)
    model = re(p)

    _, back = Zygote.pullback(model, z, x)
    vjp_z, vjp_x = back(v)
    # vjp_z = Zygote.reshape_scalar(z, Zygote.forward_jacobian(z -> re(p)(z, x), z)[2] * Zygote.vec_scalar(v))
    # NOTE: This weird sum(zero, ...) ensures that we get zeros instead of nothings
    return sum(abs2, vjp_z) / d + sum(zero, vjp_x)
end
