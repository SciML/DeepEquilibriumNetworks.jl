Flux.trainable(deq::AbstractDeepEquilibriumNetwork) = (deq.p,)

mutable struct DEQTrainingStats
    nfe::Int
end

struct DeepEquilibriumNetwork{M,P,RE,A,K} <: AbstractDeepEquilibriumNetwork
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

function (deq::DeepEquilibriumNetwork)(x::AbstractArray{T}, p = deq.p) where {T}
    deq.stats.nfe += 1
    z = deq.re(p)(zero(x), x)
    # Solving the equation f(u) - u = du = 0
    function dudt(u, _p, t)
        deq.stats.nfe += 1
        return deq.re(_p)(u, x) .- u
    end
    ssprob = SteadyStateProblem(ODEProblem(dudt, z, (zero(T), one(T)), p))
    return solve(ssprob, deq.args...; u0 = z,
                 sensealg = SteadyStateAdjoint(autodiff = false,
                                               autojacvec = ZygoteVJP()),
                 deq.kwargs...).u
end


function construct_iterator(deq::DeepEquilibriumNetwork, x, p = deq.p)
    executions = 1
    model = deq.re(p)
    previous_value = nothing
    function iterator()
        z = model(executions == 1 ? zero(x) : previous_value, x)
        executions += 1
        previous_value = z
        return z
    end
    return iterator
end


struct SkipDeepEquilibriumNetwork{M,S,P,RE1,RE2,A,K} <: AbstractDeepEquilibriumNetwork
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
    return solve(ssprob, deq.args...; u0 = z,
                 sensealg = SteadyStateAdjoint(autodiff = false,
                                               autojacvec = ZygoteVJP()),
                 deq.kwargs...).u, z
end

function construct_iterator(deq::SkipDeepEquilibriumNetwork, x, p = deq.p)
    p1, p2 = p[1:deq.split_idx], p[deq.split_idx+1:end]
    executions = 1
    model = deq.re1(p1)
    shortcut = deq.re2(p2)
    previous_value = nothing
    function iterator()
        if executions == 1
            z = shortcut(x)
        else
            z = model(previous_value, x)
        end
        executions += 1
        previous_value = z
        return z
    end
    return iterator
end