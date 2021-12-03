struct SkipDeepEquilibriumNetwork{M,S,P,RE1,RE2,A,Se,K} <: AbstractDeepEquilibriumNetwork
    model::M
    shortcut::S
    p::P
    re1::RE1
    re2::RE2
    split_idx::Int
    args::A
    kwargs::K
    sensealg::Se
    stats::DEQTrainingStats
end

Flux.@functor SkipDeepEquilibriumNetwork

function SkipDeepEquilibriumNetwork(model, shortcut, solver; p=nothing,
                                    sensealg=get_default_ssadjoint(0.1f0, 0.1f0, 10), kwargs...)
    p1, re1 = destructure(model)
    p2, re2 = destructure(shortcut)
    p = p === nothing ? vcat(p1, p2) : p
    return SkipDeepEquilibriumNetwork(model, shortcut, p, re1, re2, length(p1), (solver,), kwargs, sensealg,
                                      DEQTrainingStats(0))
end

function SkipDeepEquilibriumNetwork(model, solver; p=nothing,
                                    sensealg=SteadyStateAdjoint(; autodiff=false, autojacvec=ZygoteVJP(),
                                                                linsolve=LinSolveKrylovJL(; rtol=0.1f0, atol=0.1f0)),
                                    kwargs...)
    p1, re1 = destructure(model)
    p = p === nothing ? p1 : p
    return SkipDeepEquilibriumNetwork(model, nothing, p, re1, nothing, length(p1), (solver,), kwargs, sensealg,
                                      DEQTrainingStats(0))
end

function (deq::SkipDeepEquilibriumNetwork)(x::AbstractArray{T}, p=deq.p) where {T}
    p1, p2 = p[1:(deq.split_idx)], p[(deq.split_idx + 1):end]
    z = deq.re2(p2)(x)::typeof(x)

    # Dummy call to ensure that mask is generated
    Zygote.@ignore _ = deq.re1(p1)(x, x)

    return (solve_steady_state_problem(deq.re1, p1, x, z, deq.sensealg, deq.args...; dudt=nothing,
                                       update_nfe=() -> (deq.stats.nfe += 1), deq.kwargs...), z)
end

function (deq::SkipDeepEquilibriumNetwork{M,Nothing})(x::AbstractArray{T}, p=deq.p) where {M,T}
    z = deq.re1(p)(zero(x), x)::typeof(x)

    return (solve_steady_state_problem(deq.re1, p, x, z, deq.sensealg, deq.args...; dudt=nothing,
                                       update_nfe=() -> (deq.stats.nfe += 1), deq.kwargs...), z)
end

function (deq::SkipDeepEquilibriumNetwork)(inputs::Tuple, p=deq.p)
    # Atomic Graph Nets
    lapl, x = inputs
    p1, p2 = p[1:(deq.split_idx)], p[(deq.split_idx + 1):end]
    u0 = deq.re2(p2)(lapl, x)[2]

    function dudt(u, _p, t)
        deq.stats.nfe += 1
        return deq.re1(_p)(lapl, u, x)[2] .- u
    end

    ssprob = SteadyStateProblem(dudt, u0, p1)
    sol = solve(ssprob, deq.args...; u0=u0, sensealg=deq.sensealg, deq.kwargs...)
    deq.stats.nfe += 1

    return deq.re1(p1)(lapl, sol.u, x), u0
end

function (deq::SkipDeepEquilibriumNetwork{M,Nothing})(inputs::Tuple, p=deq.p) where {M}
    # Atomic Graph Nets
    lapl, x = inputs
    # NOTE: encoded_features being 0 causes NaN gradients to propagate
    u0 = deq.re1(p)(lapl, zero(x) .+ eps(eltype(x)), x)[2]

    function dudt(u, _p, t)
        deq.stats.nfe += 1
        return deq.re1(_p)(lapl, u, x)[2] .- u
    end

    ssprob = SteadyStateProblem(dudt, u0, p)
    sol = solve(ssprob, deq.args...; u0=u0, sensealg=deq.sensealg, deq.kwargs...)
    deq.stats.nfe += 1

    return deq.re1(p)(lapl, sol.u, x), u0
end
