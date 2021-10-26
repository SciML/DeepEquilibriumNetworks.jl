struct DEQChain{S,P1,D,P2}
    pre_deq::P1
    deq::D
    post_deq::P2

    function DEQChain(layers...)
        if length(layers) == 3
            pre_deq, deq, post_deq = layers
            is_sdeq =
                deq isa SkipDeepEquilibriumNetwork ? true :
                deq isa DeepEquilibriumNetwork ? false :
                error("$(deq) Must be a DEQ or SkipDEQ")
            return new{is_sdeq,typeof(pre_deq),typeof(deq),typeof(post_deq)}(
                pre_deq,
                deq,
                post_deq,
            )
        end
        pre_deq = []
        post_deq = []
        deq = nothing
        encounter_deq = false
        is_sdeq = false
        for l in layers
            if l isa SkipDeepEquilibriumNetwork || l isa DeepEquilibriumNetwork
                is_sdeq = l isa SkipDeepEquilibriumNetwork
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
        return new{is_sdeq,typeof(pre_deq),typeof(deq),typeof(post_deq)}(
            pre_deq,
            deq,
            post_deq,
        )
    end
end

Flux.@functor DEQChain

(deq::DEQChain{false})(x) = deq.post_deq(deq.deq(deq.pre_deq(x)))

function (deq::DEQChain{true})(x)
    x1 = deq.pre_deq(x)
    z, ẑ = deq.deq(x1)
    x2 = deq.post_deq(z)
    return (x2, (z, ẑ))
end

function get_and_clear_nfe!(model::DEQChain)
    nfe = model.deq.stats.nfe
    model.deq.stats.nfe = 0
    return nfe
end
