# Doesn't work as of now
function compute_deq_jacobian_loss(
    model, ps::ComponentArray, st::NamedTuple, z::AbstractArray, x::AbstractArray
)
    l, back = Zygote.pullback(u -> model((u, x), ps, st)[1], z)
    vjp_z = back(gaussian_like(l))[1]
    return sum(abs2, vjp_z) / length(z)
end
