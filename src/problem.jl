# No need for this file. We can work around the Zygote gradient drop bug by treating
# the input as a parameter.

struct DEQSteadyStateProblem{uType,isinplace,P,F,X,K} <:
       SciMLBase.AbstractSteadyStateProblem{uType,isinplace}
    # f(u, x, p, t)  --> x is already collapsed so we only have 3 parameters but we need the gradient wrt x
    f::F
    u0::uType
    p::P
    x::X
    kwargs::K
    SciMLBase.@add_kwonly function DEQSteadyStateProblem{iip}(
        f::SciMLBase.AbstractODEFunction{iip},
        u0,
        p,
        x;
        kwargs...,
    ) where {iip}
        return new{
            typeof(u0),
            isinplace(f),
            typeof(p),
            typeof(f),
            typeof(x),
            typeof(kwargs),
        }(
            f,
            u0,
            p,
            x,
            kwargs,
        )
    end
end

function DEQSteadyStateProblem(
    f::SciMLBase.AbstractODEFunction,
    u0,
    p,
    x;
    kwargs...,
)
    return DEQSteadyStateProblem{SciMLBase.isinplace(f)}(f, u0, p, x; kwargs...)
end

function DEQSteadyStateProblem(f, u0, p, x; kwargs...)
    return DEQSteadyStateProblem(ODEFunction(f), u0, p, x; kwargs...)
end

function DEQSteadyStateProblem(prob::SciMLBase.AbstractODEProblem, x)
    return DEQSteadyStateProblem{SciMLBase.isinplace(prob)}(
        prob.f,
        prob.u0,
        prob.p,
        x,
    )
end

function SciMLBase.__solve(
    prob::DEQSteadyStateProblem,
    alg::DynamicSS,
    x;
    save_everystep = false,
    save_start = false,
    save_idxs = nothing,
    kwargs...,
)
    tspan =
        alg.tspan isa Tuple ? alg.tspan :
        convert.(real(eltype(prob.u0)), (zero(alg.tspan), alg.tspan))
    _prob = ODEProblem{false}(
        (u, p, t) -> prob.f(u, x, p, t),
        prob.u0,
        tspan,
        prob.p,
    )
    sol = solve(
        _prob,
        alg.alg;
        kwargs...,
        callback = TerminateSteadyState(alg.abstol, alg.reltol),
        save_everystep = save_everystep,
        save_start = save_start,
    )

    du = prob.f(sol.u[end], x, prob.p, sol.t[end])

    array_condition() = all(
        abs(d) <= abstol || abs(d) <= reltol * abs(u) for
        (d, abstol, reltol, u) in zip(
            du,
            Iterators.cycle(alg.abstol),
            Iterators.cycle(alg.reltol),
            sol.u[end],
        )
    )
    broadcast_condition() = all(
        (abs.(du) .<= alg.abstol) .|
        (abs.(du) .<= alg.reltol .* abs.(sol.u[end])),
    )

    if save_idxs !== nothing
        u = sol.u[end][save_idxs]
        du = du[save_idxs]
    else
        u = sol.u[end]
    end

    if sol.retcode == :Terminated &&
       (typeof(sol.u[end]) <: Array ? array_condition() : broadcast_condition())
        _sol = DiffEqBase.build_solution(prob, alg, u, du; retcode = :Success)
    else
        _sol = DiffEqBase.build_solution(prob, alg, u, du; retcode = :Failure)
    end

    return _sol
end

# TODO
# function SciMLBase.__solve(
#     prob::DEQSteadyStateProblem,
#     alg::SSRootfind;
#     kwargs...,
# )
#     _prob = SteadyStateProblem{false}(
#         (u, p, t) -> prob.f(u, prob.x, p, t),
#         prob.u0,
#         prob.p,
#     )
#     return SciMLBase.__solve(_prob, alg, prob.x; kwargs...)
# end

function DiffEqBase.get_concrete_problem(
    prob::DEQSteadyStateProblem,
    isadapt;
    kwargs...,
)
    u0 = DiffEqBase.get_concrete_u0(prob, isadapt, Inf, kwargs)
    u0 = DiffEqBase.promote_u0(u0, prob.p, nothing)
    return remake(prob; u0 = u0)
end


function DiffEqBase._concrete_solve_adjoint(
    prob::DEQSteadyStateProblem,
    alg,
    sensealg::SteadyStateAdjoint,
    u0,
    p,
    x;
    save_idxs = nothing,
    kwargs...,
)
    _prob = remake(prob, u0 = u0, p = p)
    sol = solve(_prob, alg, x; kwargs...)
    _save_idxs = save_idxs === nothing ? Colon() : save_idxs

    if save_idxs === nothing
        out = sol
    else
        out = DiffEqBase.sensitivity_solution(sol, sol[_save_idxs])
    end

    function deqsteadystatebackpass(Δ)
        dp, dx = deq_adjoint_sensitivities(
            sol,
            x,
            sensealg,
            Δ;
            save_idxs = save_idxs,
        )
        return (
            NoTangent(),
            NoTangent(),
            NoTangent(),
            NoTangent(),
            dp,
            NoTangent(),
            dx,
        )
    end
    return out, deqsteadystatebackpass
end


function deq_adjoint_sensitivities(sol, x, sensealg, dg; save_idxs)
    @unpack f, p = sol.prob
    return p, x
end
