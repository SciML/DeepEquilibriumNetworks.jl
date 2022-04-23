struct DeepEquilibriumNetwork{J,M,A,S,K} <: AbstractDeepEquilibriumNetwork
    model::M
    solver::A
    sensealg::S
    kwargs::K
end

function DeepEquilibriumNetwork(
    model, solver; jacobian_regularization::Bool=false, sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10), kwargs...
)
    return DeepEquilibriumNetwork{jacobian_regularization,typeof(model),typeof(solver),typeof(sensealg),typeof(kwargs)}(
        model, solver, sensealg, kwargs
    )
end

function (deq::DeepEquilibriumNetwork{J})(
    x::AbstractArray{T}, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
) where {J,T}
    z = zero(x)

    if !iszero(st.fixed_depth)
        # Pretraining without Fixed Point Solving
        st_ = st.model
        z_star = z
        for _ in 1:(st.fixed_depth)
            z_star, st_ = deq.model((z_star, x), ps, st_)
        end
        @set! st.model = st_

        return (z_star, DeepEquilibriumSolution(z_star, z, z, 0.0f0, st.fixed_depth)), st
    end

    st_ = st.model

    function dudt(u, p, t)
        u_, st_ = deq.model((u, x), p, st_)
        return u_ .- u
    end

    prob = SteadyStateProblem(ODEFunction{false}(dudt), z, ps)
    sol = solve(prob, deq.solver; sensealg=deq.sensealg, deq.kwargs...)
    z_star, st_ = deq.model((sol.u, x), ps, st.model)

    jac_loss = (J ? compute_deq_jacobian_loss(deq.model, ps, st.model, z_star, x) : T(0))
    residual = z_star .- deq.model((z_star, x), ps, st.model)[1]

    st_ = EFL.update_state(st_, :update_mask, true)
    @set! st.model = st_

    return (z_star, DeepEquilibriumSolution(z_star, z, residual, jac_loss, sol.destats.nf + 1 + J)), st
end

struct SkipDeepEquilibriumNetwork{J,M,Sh,A,S,K} <: AbstractSkipDeepEquilibriumNetwork
    model::M
    shortcut::Sh
    solver::A
    sensealg::S
    kwargs::K
end

function SkipDeepEquilibriumNetwork(
    model,
    shortcut,
    solver;
    jacobian_regularization::Bool=false,
    sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10),
    kwargs...,
)
    return SkipDeepEquilibriumNetwork{
        jacobian_regularization,typeof(model),typeof(shortcut),typeof(solver),typeof(sensealg),typeof(kwargs)
    }(
        model, shortcut, solver, sensealg, kwargs
    )
end

function (deq::SkipDeepEquilibriumNetwork{J,M,S})(
    x::AbstractArray{T}, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
) where {J,M,S,T}
    z, st__ = if S == Nothing
        deq.model((zero(x), x), ps.model, st.model)
    else
        deq.shortcut(x, ps.shortcut, st.shortcut)
    end
    @set! st.shortcut = st__

    if !iszero(st.fixed_depth)
        # Pretraining without Fixed Point Solving
        st_ = st.model
        z_star = z
        for _ in 1:(st.fixed_depth)
            z_star, st_ = deq.model((z_star, x), ps.model, st_)
        end
        @set! st.model = st_

        return (z_star, DeepEquilibriumSolution(z_star, z, z, 0.0f0, st.fixed_depth)), st
    end

    st_ = st.model

    function dudt(u, p, t)
        u_, st_ = deq.model((u, x), p, st_)
        return u_ .- u
    end

    prob = SteadyStateProblem(ODEFunction{false}(dudt), z, ps.model)
    sol = solve(prob, deq.solver; sensealg=deq.sensealg, deq.kwargs...)
    z_star, st_ = deq.model((sol.u, x), ps.model, st.model)

    jac_loss = (J ? compute_deq_jacobian_loss(deq.model, ps.model, st.model, z_star, x) : T(0))
    residual = z_star .- deq.model((z_star, x), ps.model, st.model)[1]

    st_ = EFL.update_state(st_, :update_mask, true)
    @set! st.model = st_

    return (z_star, DeepEquilibriumSolution(z_star, z, residual, jac_loss, sol.destats.nf + 1 + J)), st
end
