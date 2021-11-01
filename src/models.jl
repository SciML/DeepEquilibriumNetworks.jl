modeltype_to_val(::DeepEquilibriumNetwork) = Val(1)
modeltype_to_val(::SkipDeepEquilibriumNetwork) = Val(2)
modeltype_to_val(::MultiScaleDeepEquilibriumNetworkS4) = Val(3)
modeltype_to_val(::MultiScaleSkipDeepEquilibriumNetworkS4) = Val(4)
modeltype_to_val(m) = Val(-1)

struct DEQChain{V,P1,D,P2}
    pre_deq::P1
    deq::D
    post_deq::P2

    function DEQChain(layers...)
        if length(layers) == 3
            pre_deq, deq, post_deq = layers
            val = modeltype_to_val(deq)
            val == Val(-1) && error(
                "$deq must subtype AbstractDeepEquilibriumNetwork and define `modeltype_to_val`",
            )
            return new{val,typeof(pre_deq),typeof(deq),typeof(post_deq)}(
                pre_deq,
                deq,
                post_deq,
            )
        end
        pre_deq = []
        post_deq = []
        deq = nothing
        encounter_deq = false
        global_val = Val(1)
        for l in layers
            val = modeltype_to_val(l)
            if val != Val(-1)
                global_val = val
                encounter_deq &&
                    error("Can have only 1 DEQ Layer in the Chain!!!")
                deq = l
                encounter_deq = true
                continue
            end
            if encounter_deq
                push!(post_deq, l)
            else
                push!(pre_deq, l)
            end
        end
        !encounter_deq &&
            error("No DEQ Layer in the Chain!!! Maybe you wanted to use Chain")
        pre_deq = length(pre_deq) == 0 ? identity : Chain(pre_deq...)
        post_deq = length(post_deq) == 0 ? identity : Chain(post_deq...)
        return new{global_val,typeof(pre_deq),typeof(deq),typeof(post_deq)}(
            pre_deq,
            deq,
            post_deq,
        )
    end
end

Flux.@functor DEQChain

(deq::Union{DEQChain{Val(1)}, DEQChain{Val(3)}})(x) = deq.post_deq(deq.deq(deq.pre_deq(x)))

function (deq::DEQChain{Val(2)})(x)
    x1 = deq.pre_deq(x)
    z, ẑ = deq.deq(x1)
    x2 = deq.post_deq(z)
    return (x2, (z, ẑ))
end

function (deq::DEQChain{Val(4)})(x)
    x1 = deq.pre_deq(x)
    z, ẑ = deq.deq(x1)
    x2 = deq.post_deq(z)
    return (x2, tuple(zip(z, ẑ)...))
end

function get_and_clear_nfe!(model::DEQChain)
    nfe = model.deq.stats.nfe
    model.deq.stats.nfe = 0
    return nfe
end
