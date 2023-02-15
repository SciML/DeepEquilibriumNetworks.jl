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
  sol = DiffEqBase.__solve(prob, alg.alg, args...; kwargs...)

  u, du = sol.u, sol.resid
  uType = typeof(u)

  # TODO: Allow propagation of destats in NonlinearSolution?
  destats = (nf=-1,)

  return EquilibriumSolution{eltype(uType), ndims(uType), uType, typeof(prob), typeof(alg),
                             typeof(destats)}(u, du, prob, alg, retcode, destats)
end

# ========================================
# TODO(@avik-pal): Very Temporary Solution. Remove before Merging!!!
struct _FakeIntegrator{DU, U}
  du::DU
  u::U
end

DiffEqBase.get_du(integrator::_FakeIntegrator) = integrator.du

function _get_terminate_condition(alg::DiscreteDEQSolver, args...; kwargs...)
  tc = alg.termination_condition
  termination_condition = _get_termination_condition(tc, args...; kwargs...)
  function _termination_condition_closure_discrete_deq(du, u)
    return termination_condition(_FakeIntegrator(du, u), tc.abstol, tc.reltol, nothing)
  end
  return _termination_condition_closure_discrete_deq
end
# ========================================

function DiffEqBase.__solve(prob::AbstractSteadyStateProblem{uType}, alg::DiscreteDEQSolver,
                            args...; maxiters=10, kwargs...) where {uType}
  terminate_stats = Dict{Symbol, Any}(:best_objective_value => real(eltype(prob.u0))(Inf),
                                      :best_objective_value_iteration => nothing)

  us, stats = nlsolve(alg.alg, u -> prob.f(u, prob.p, nothing), prob.u0; maxiters,
                      terminate_condition=_get_terminate_condition(alg, terminate_stats))

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
