module DeepEquilibriumNetworksZygoteExt

using ADTypes, Statistics, Zygote
import DeepEquilibriumNetworks: __gaussian_like, __estimate_jacobian_trace

function __estimate_jacobian_trace(::AutoZygote, model, ps, z, x, rng)
    res, back = Zygote.pullback(u -> model((u, x), ps), z)
    vjp_z = only(back(__gaussian_like(rng, res)))
    return mean(abs2, vjp_z)
end

end
