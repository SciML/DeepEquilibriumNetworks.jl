struct EquilibriumSolution{T,N,uType,P,A,D} <: SciMLBase.AbstractNonlinearSolution{T,N}
    u::uType
    resid::uType
    prob::P
    alg::A
    retcode::Symbol
    destats::D
end

function transform_solution(soln::EquilibriumSolution)
    # Creates a NonlinearSolution/SteadyStateSolution
    return DiffEqBase.build_solution(soln.prob, soln.alg, soln.u, soln.resid; retcode=soln.retcode)
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractSteadyStateProblem{uType}, alg::ContinuousDEQSolver, args...; kwargs...) where {uType}
    tspan = alg.tspan isa Tuple ? alg.tspan : convert.(real(eltype(prob.u0)), (zero(alg.tspan), alg.tspan))
    _prob = ODEProblem(prob.f, prob.u0, tspan, prob.p)

    terminate_stats = Dict{Symbol,Any}(:best_objective_value => real(eltype(prob.u0))(Inf),
                                       :best_objective_value_iteration => nothing)

    sol = solve(_prob, alg.alg, args...; kwargs...,
                callback=TerminateSteadyState(alg.abstol, alg.reltol, get_terminate_condition(alg, terminate_stats)))

    u, t = terminate_stats[:best_objective_value_iteration] === nothing ? (sol.u[end], sol.t[end]) :
           (sol.u[terminate_stats[:best_objective_value_iteration] + 1],
            sol.t[terminate_stats[:best_objective_value_iteration] + 1])

    # Dont count towards NFE since this is mostly a check for convergence
    du = prob.f(u, prob.p, t)

    retcode = (sol.retcode == :Terminated && has_converged(du, u, alg) ? :Success : :Failure)

    return EquilibriumSolution{eltype(uType),ndims(uType),uType,typeof(prob),typeof(alg),typeof(sol.destats)}(u, du, prob, alg, retcode, sol.destats)
end
