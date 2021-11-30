struct DeepEquilibriumNetwork{J,M,P,RE,A,S,K} <: AbstractDeepEquilibriumNetwork
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
        deq.kwargs...,
    )
end

function DeepEquilibriumNetwork(
    model,
    args...;
    jacobian_regularization::Bool = false,
    p = nothing,
    sensealg = get_default_ssadjoint(0.1f0, 0.1f0, 10),
    kwargs...,
)
    _p, re = Flux.destructure(model)
    p = p === nothing ? _p : p
    stats = DEQTrainingStats(0)
    return DeepEquilibriumNetwork{
        jacobian_regularization,
        typeof(model),
        typeof(p),
        typeof(re),
        typeof(args),
        typeof(kwargs),
        typeof(sensealg),
        typeof(stats),
    }(
        model,
        p,
        re,
        args,
        kwargs,
        sensealg,
        stats,
    )
end

function (deq::DeepEquilibriumNetwork)(x::AbstractArray{T}, p = deq.p) where {T}
    # Solving the equation f(u) - u = du = 0
    z = deq.re(p)(zero(x), x)::typeof(x)
    deq.stats.nfe += 1

    update_is_mask_reset_allowed(false)

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

    res = deq.re(p)(sol.u, x)::typeof(x)
    deq.stats.nfe += 1

    update_is_mask_reset_allowed(true)

    return res
end

function (deq::DeepEquilibriumNetwork)(inputs::Tuple, p = deq.p) where {T}
    lapl, x = inputs
    # Atomic Graph Nets
    u0 = zero(x)

    function dudt(u, _p, t)
        deq.stats.nfe += 1
        return deq.re(_p)(lapl, u, x)[2] .- u
    end

    ssprob = SteadyStateProblem(dudt, u0, p)
    sol = solve(
        ssprob,
        deq.args...;
        u0 = u0,
        sensealg = deq.sensealg,
        deq.kwargs...,
    )
    deq.stats.nfe += 1

    return deq.re(p)(lapl, sol.u, x)
end
