module DeepEquilibriumNetworksZygoteExt

using ADTypes: AutoZygote
using ChainRulesCore: ChainRulesCore
using DeepEquilibriumNetworks: DEQs
using FastClosures: @closure
using ForwardDiff: ForwardDiff  # This is a dependency of Zygote
using Lux: Lux, StatefulLuxLayer
using Statistics: mean
using Zygote: Zygote

const CRC = ChainRulesCore

@inline __tupleify(x) = @closure(u->(u, x))

## One day we will overload DI's APIs for Lux Layers and we can remove this
## Main challenge with overloading Zygote.pullback is that we need to return the correct
## tangent for the pullback to compute the correct gradient, which is quite hard. But
## wrapping the overall vjp is not that hard.
@inline function __compute_vector_jacobian_product(model::StatefulLuxLayer, ps, z, x, rng)
    res, back = Zygote.pullback(model ∘ __tupleify(x), z)
    return only(back(DEQs.__gaussian_like(rng, res)))
end

function CRC.rrule(
        ::typeof(__compute_vector_jacobian_product), model::StatefulLuxLayer, ps, z, x, rng)
    res, back = Zygote.pullback(model ∘ __tupleify(x), z)
    ε = DEQs.__gaussian_like(rng, res)
    y = only(back(ε))
    ∇internal_gradient_capture = Δ -> begin
        (Δ isa CRC.NoTangent || Δ isa CRC.ZeroTangent) &&
            return ntuple(Returns(CRC.NoTangent()), 6)

        Δ_ = reshape(CRC.unthunk(Δ), size(z))

        Tag = typeof(ForwardDiff.Tag(model, eltype(z)))
        partials = ForwardDiff.Partials{1, eltype(z)}.(tuple.(Δ_))
        z_dual = ForwardDiff.Dual{Tag, eltype(z), 1}.(z, partials)

        _, pb_f = Zygote.pullback((x1, x2, p) -> model((x1, x2), p), z_dual, x, ps)
        ∂z_duals, ∂x_duals, ∂ps_duals = pb_f(ε)

        ∂z = Lux.__partials(Tag, ∂z_duals, 1)
        ∂x = Lux.__partials(Tag, ∂x_duals, 1)
        ∂ps = Lux.__partials(Tag, ∂ps_duals, 1)

        return CRC.NoTangent(), CRC.NoTangent(), ∂ps, ∂z, ∂x, CRC.NoTangent()
    end
    return y, ∇internal_gradient_capture
end

## Don't remove `ad`. See https://github.com/ericphanson/ExplicitImports.jl/issues/33
function DEQs.__estimate_jacobian_trace(ad::AutoZygote, model, z, x, rng)
    return mean(abs2, __compute_vector_jacobian_product(model, model.ps, z, x, rng))
end

end
