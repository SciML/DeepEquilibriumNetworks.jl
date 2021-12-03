#= Testing Code
using FastDEQ, Flux

model = DeepEquilibriumNetwork(
    Parallel(
        +,
        Dense(2, 2; bias = false),
        Dense(2, 2; bias = false)
    ),
    get_default_dynamicss_solver(0.1f0, 0.1f0),
    sensealg = get_default_ssadjoint(0.1f0, 0.1f0, 10)
) |> gpu

x = rand(Float32, 2, 1) |> gpu

z = model(x)

FastDEQ.compute_deq_jacobian_loss(model.re, model.p, z, x)

Flux.gradient(p -> FastDEQ.compute_deq_jacobian_loss(model.re, p, z, x), model.p)
=#

gaussian_like(p::Array) = randn(eltype(p), size(p))
gaussian_like(p::CuArray) = CUDA.randn(eltype(p), size(p))

Zygote.@nograd gaussian_like

# NOTE: If the model internally uses destructure/restructure eg. WeightNorm Layer, then
#       this loss function will error out in the backward pass.
function compute_deq_jacobian_loss(re, p::AbstractVector{T}, z::A, x::A) where {T,A<:AbstractArray}
    d = length(z)
    v = gaussian_like(z)
    model = re(p)

    _, back = Zygote.pullback(model, z, x)
    vjp_z, vjp_x = back(v)
    # NOTE: This weird sum(zero, ...) ensures that we get zeros instead of nothings
    return (norm(vjp_z, 2)^2) / d + sum(zero, vjp_x)
end
