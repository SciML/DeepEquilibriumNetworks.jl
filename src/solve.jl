"""
    EquilibriumSolution

Wraps the solution of a SteadyStateProblem using either ContinuousDEQSolver or
DiscreteDEQSolver. This is mostly an internal implementation detail, which allows proper
dispatch during adjoint computation without type piracy.
"""
struct EquilibriumSolution{T, N, uType, P, A, D} <: AbstractNonlinearSolution{T, N}
  u::uType
  resid::uType
  prob::P
  alg::A
  retcode::Symbol
  destats::D
end

function DiffEqBase.__solve(prob::AbstractSteadyStateProblem, alg::ContinuousDEQSolver,
                            args...; kwargs...)
  sol = solve(prob, alg.alg, args...; kwargs...)

  u, du = sol.u, sol.resid
  uType = typeof(u)

  # TODO: Allow propagation of destats in NonlinearSolution?
  destats = (nf=-1,)

  return EquilibriumSolution{eltype(uType), ndims(uType), uType, typeof(prob), typeof(alg),
                             typeof(destats)}(u, du, prob, alg, retcode, destats)
end

function DiffEqBase.__solve(prob::AbstractSteadyStateProblem{uType}, alg::DiscreteDEQSolver,
                            args...; maxiters=10, kwargs...) where {uType}
  terminate_stats = Dict{Symbol, Any}(:best_objective_value => real(eltype(prob.u0))(Inf),
                                      :best_objective_value_iteration => nothing)

  us, stats = nlsolve(alg.alg, u -> prob.f(u, prob.p, nothing), prob.u0; maxiters,
                      terminate_condition=alg.termination_condition(terminate_stats),
                      abstol=alg.termination_condition.abstol,
                      reltol=alg.termination_condition.reltol)

  u = if terminate_stats[:best_objective_value_iteration] === nothing
    us[end]
  else
    us[terminate_stats[:best_objective_value_iteration] + 1]
  end

  # Dont count towards NFE since this is mostly a check for convergence
  du = prob.f(u, prob.p, nothing)

  retcode = has_converged(du, u, alg) ? :Success : :Failure

  destats = (nf=stats.nf,)

  return EquilibriumSolution{eltype(uType), ndims(uType), uType, typeof(prob), typeof(alg),
                             typeof(destats)}(u, du, prob, alg, retcode, destats)
end
