# For testing purposes atm
struct DEQSolver{A,AT,RT,TS} <: SteadyStateDiffEq.SteadyStateDiffEqAlgorithm
    alg::A
    abstol::AT
    reltol::RT
    tspan::TS
end

DEQSolver(alg; abstol=1e-8, reltol=1e-8, tspan=Inf) = DEQSolver(alg, abstol, reltol, tspan)

# For DEQs we typically dont care about the abstol
function terminate_condition(integrator, abstol, reltol)
    return all(abs.(DiffEqBase.get_du(integrator)) .<= reltol .* abs.(integrator.u))
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractSteadyStateProblem, alg::DEQSolver, args...; kwargs...)
    tspan = alg.tspan isa Tuple ? alg.tspan : convert.(real(eltype(prob.u0)), (zero(alg.tspan), alg.tspan))
    _prob = ODEProblem(prob.f, prob.u0, tspan, prob.p)
    sol = solve(_prob, alg.alg, args...; kwargs...,
                callback=TerminateSteadyState(alg.abstol, alg.reltol, terminate_condition))

    du = prob.f(sol.u[end], prob.p, sol.t[end])

    if sol.retcode == :Terminated && all(abs.(du) .<= alg.reltol .* abs.(sol.u[end]))
        return DiffEqBase.build_solution(prob, alg, sol.u[end], du; retcode=:Success)
    else
        return DiffEqBase.build_solution(prob, alg, sol.u[end], du; retcode=:Failure)
    end
end
