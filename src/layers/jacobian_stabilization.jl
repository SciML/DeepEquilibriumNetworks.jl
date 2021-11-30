gaussian_like(p::Array) = randn(eltype(p), size(p))
gaussian_like(p::CuArray) = CUDA.randn(eltype(p), size(p))

Zygote.@nograd gaussian_like

# FIXME: Not yet functional
function compute_deq_jacobian_loss(
    re,
    p::AbstractVector{T},
    z::A,
    x::A,
) where {T,A<:AbstractArray}
    d = length(z)
    v = gaussian_like(z)
    # model = Zygote.@showgrad re(p)

    # Zygote over Zygote doesn't work :(
    _, back = Zygote.pullback(p -> re(p)(z, x), p)
    return (norm(back(v), 2)^2) / d

    # f0 = f(p)
    # ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(p)))
    # for i ∈ 1:length(p)

    # end
end
