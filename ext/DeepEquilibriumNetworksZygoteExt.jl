module DeepEquilibriumNetworksZygoteExt

using ADTypes: AutoZygote
using FastClosures: @closure
using Statistics: mean
using Zygote: Zygote
using DeepEquilibriumNetworks: DEQs

@inline __tupleify(u) = @closure x -> (u, x)

## Don't remove `ad`. See https://github.com/ericphanson/ExplicitImports.jl/issues/33
## FIXME: This will be broken in the new Lux release let's fix this
function DEQs.__estimate_jacobian_trace(ad::AutoZygote, model, z, x, rng)
    res, back = Zygote.pullback(model ∘ __tupleify, z)
    vjp_z = only(back(DEQs.__gaussian_like(rng, res)))
    return mean(abs2, vjp_z)
end

end
