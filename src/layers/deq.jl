struct DeepEquilibriumNetwork{M,P,RE,A,S,K} <: AbstractDeepEquilibriumNetwork
    model::M
    p::P
    re::RE
    args::A
    kwargs::K
    sensealg::S
    stats::DEQTrainingStats
end

Flux.@functor DeepEquilibriumNetwork

function Flux.gpu(deq::DeepEquilibriumNetwork)
    return DeepEquilibriumNetwork(
        deq.model |> gpu,
        deq.args...;
        p = deq.p |> gpu,
        sensealg = deq.sensealg,
        deq.kwargs...
    )
end

function DeepEquilibriumNetwork(
    model,
    args...;
    p = nothing,
    sensealg = SteadyStateAdjoint(
        autodiff = false,
        autojacvec = ZygoteVJP(),
        linsolve = LinSolveKrylovJL(rtol = 0.1f0, atol = 0.1f0),
    ),
    kwargs...,
)
    _p, re = Flux.destructure(model)
    p = p === nothing ? _p : p
    return DeepEquilibriumNetwork(
        model,
        p,
        re,
        args,
        kwargs,
        sensealg,
        DEQTrainingStats(0),
    )
end

function (deq::DeepEquilibriumNetwork)(x::AbstractArray{T}, p = deq.p) where {T}
    # Solving the equation f(u) - u = du = 0
    z = zero(x)

    function dudt(u, _p, t)
        deq.stats.nfe += 1
        return deq.re(_p)(u, x) .- u
    end

    ssprob = SteadyStateProblem(dudt, z, p)
    sol = solve(
        ssprob,
        deq.args...;
        u0 = z,
        sensealg = deq.sensealg,
        deq.kwargs...,
    )
    deq.stats.nfe += 1
    return deq.re(p)(sol.u, x)::typeof(x)
end

# function (deq::DeepEquilibriumNetwork)(lapl::AbstractMatrix{T}, x::AbstractArray{T}, p = deq.p) where {T}
#     # Atomic Graph Nets
#     u0 = zero(x)

#     function dudt(u, _p, t)
#         deq.stats.nfe += 1
#         return deq.re(_p)(lapl, u, x)[2] .- u
#     end

#     ssprob = SteadyStateProblem(dudt, u0, p)
#     sol = solve(
#         ssprob,
#         deq.args...;
#         u0 = u0,
#         sensealg = deq.sensealg,
#         deq.kwargs...,
#     )
#     deq.stats.nfe += 1

#     return deq.re(p)(lapl, sol.u, x)
# end