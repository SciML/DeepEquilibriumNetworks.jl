# Default to nothing happening
reset_mask!(x) = nothing

"""
    DEQChain(pre_deq, deq, post_deq)
    DEQChain(layers...)

A Sequential Model containing a DEQ.

!!! note
    The Model should contain exactly 1 `AbstractDEQ` Layer
"""
struct DEQChain{P1,D<:AbstractDeepEquilibriumNetwork,P2}
    pre_deq::P1
    deq::D
    post_deq::P2
end

function DEQChain(layers...)
    pre_deq, post_deq, deq, encounter_deq = [], [], nothing, false
    for l in layers
        if typeof(l) <: AbstractDeepEquilibriumNetwork
            @assert !encounter_deq "Can have only 1 DEQ Layer in the Chain!!!"
            deq = l
            encounter_deq = true
            continue
        end
        push!(encounter_deq ? post_deq : pre_deq, l)
    end
    @assert encounter_deq "No DEQ Layer in the Chain!!! Maybe you wanted to use Chain"
    pre_deq = length(pre_deq) == 0 ? NoOpLayer() : (length(pre_deq) == 1 ? pre_deq[1] : FChain(pre_deq...))
    post_deq = length(post_deq) == 0 ? NoOpLayer() : (length(post_deq) == 1 ? post_deq[1] : FChain(post_deq...))
    return DEQChain(pre_deq, deq, post_deq)
end

Flux.@functor DEQChain

function (deq::DEQChain)(x; kwargs...)
    x1 = deq.pre_deq(x)
    x2, deq_soln = deq.deq(x1; kwargs...)
    x3 = deq.post_deq(x2)
    return x3, deq_soln
end

function get_and_clear_nfe!(model::DEQChain)
    nfe = model.deq.stats.nfe
    model.deq.stats.nfe = 0
    return nfe
end

function Base.show(io::IO, model::DEQChain)
    l1 = length(destructure_parameters(model)[1])
    println(io, "DEQChain(")
    print(io, "\t")
    show(io, model.pre_deq)
    print(io, "\n\t")
    show(io, model.deq)
    print(io, "\n\t")
    show(io, model.post_deq)
    return print(io, "\n) $l1 Trainable Parameters")
end
