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
    # NOTE: This weird sum(zero, ...) ensures that we get zeros instead of nothings
    return sum(abs2, vjp_z) / d + sum(zero, vjp_x)
end

function compute_deq_jacobian_loss(re, p::AbstractVector{T}, lapl::A, z::A, x::A) where {T,A<:AbstractArray}
    d = length(z)
    v = gaussian_like(z)
    model = re(p)

    ## FIXME: Doesn't work as of now...
    _, back = Zygote.pullback((lapl, z, x) -> model(lapl, z, x)[2], lapl, z, x)
    vjp_lapl, vjp_z, vjp_x = back(v)
    # NOTE: This weird sum(zero, ...) ensures that we get zeros instead of nothings
    return sum(abs2, vjp_z) / d + sum(zero, vjp_x) + sum(zero, vjp_lapl)
end
