mutable struct DEQTrainingStats
    nfe::Int
end

struct DeepEquilibriumNetwork{M,P,RE,A,K}
    model::M
    p::P
    re::RE
    args::A
    kwargs::K
    stats::DEQTrainingStats
end

Flux.@functor DeepEquilibriumNetwork

function DeepEquilibriumNetwork(model, args...; p = nothing, kwargs...)
    _p, re = Flux.destructure(model)
    p = p === nothing ? _p : p
    return DeepEquilibriumNetwork(
        model,
        p,
        re,
        args,
        kwargs,
        DEQTrainingStats(0),
    )
end

Flux.trainable(deq::DeepEquilibriumNetwork) = (deq.p,)

function (deq::DeepEquilibriumNetwork)(x::AbstractArray{T}, p = deq.p) where {T}
    deq.stats.nfe += 1
    z = deq.re(p)(zero(x), x)
    # Solving the equation f(u) - u = du = 0
    function dudt(u, _p, t)
        deq.stats.nfe += 1
        return deq.re(_p)(u, x) .- u
    end
    ssprob = SteadyStateProblem(ODEProblem(dudt, z, (zero(T), one(T)), p))
    return solve(ssprob, deq.args...; u0 = z, deq.kwargs...).u
end


struct SkipDeepEquilibriumNetwork{M,S,P,RE1,RE2,A,K}
    model::M
    shortcut::S
    p::P
    re1::RE1
    re2::RE2
    split_idx::Int
    args::A
    kwargs::K
    stats::DEQTrainingStats
end

Flux.@functor SkipDeepEquilibriumNetwork

function SkipDeepEquilibriumNetwork(
    model,
    shortcut,
    args...;
    p = nothing,
    kwargs...,
)
    p1, re1 = Flux.destructure(model)
    p2, re2 = Flux.destructure(shortcut)
    p = p === nothing ? vcat(p1, p2) : p
    return SkipDeepEquilibriumNetwork(
        model,
        shortcut,
        p,
        re1,
        re2,
        length(p1),
        args,
        kwargs,
        DEQTrainingStats(0),
    )
end

Flux.trainable(deq::SkipDeepEquilibriumNetwork) = (deq.p,)

function (deq::SkipDeepEquilibriumNetwork)(
    x::AbstractArray{T},
    p = deq.p,
) where {T}
    p1, p2 = p[1:deq.split_idx], p[deq.split_idx+1:end]
    z = deq.re2(p2)(x)
    deq.stats.nfe += 1
    # Solving the equation f(u) - u = du = 0
    function dudt(u, _p, t)
        deq.stats.nfe += 1
        return deq.re1(_p)(u, x) .- u
    end
    ssprob = SteadyStateProblem(ODEProblem(dudt, z, (zero(T), one(T)), p1))
    return solve(ssprob, deq.args...; u0 = z, deq.kwargs...).u, z
end
