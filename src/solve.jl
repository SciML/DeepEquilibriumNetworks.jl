import DiffEqCallbacks
import DiffEqBase
import OrdinaryDiffEq
import SciMLBase

struct UnrolledDEQSolution{T, N, uType, D} <: SciMLBase.AbstractNonlinearSolution{T, N}
  u::uType
  resid::uType
  destats::D
end

function UnrolledDEQSolution(u, resid, destats)
  return UnrolledDEQSolution{eltype(u), ndims(u), typeof(u), typeof(destats)}(u, resid,
                                                                              destats)
end

"""
    EquilibriumSolution

Wraps the solution of a SteadyStateProblem using either ContinuousDEQSolver or
DiscreteDEQSolver. This is mostly an internal implementation detail, which allows proper
dispatch during adjoint computation without type piracy.
"""
struct EquilibriumSolution{T, N, uType, P, A, D, S} <:
       SciMLBase.AbstractNonlinearSolution{T, N}
  u::uType
  resid::uType
  prob::P
  alg::A
  retcode::Symbol
  destats::D
  sol::S
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractSteadyStateProblem{uType},
                            alg::ContinuousDEQSolver, args...; kwargs...) where {uType}
  tspan = alg.tspan isa Tuple ? alg.tspan :
          convert.(real(eltype(prob.u0)), (zero(alg.tspan), alg.tspan))
  _prob = OrdinaryDiffEq.ODEProblem(prob.f, prob.u0, tspan, prob.p)

  terminate_stats = Dict{Symbol, Any}(:best_objective_value => real(eltype(prob.u0))(Inf),
                                      :best_objective_value_iteration => nothing)

  callback = DiffEqCallbacks.TerminateSteadyState(alg.abstol_termination,
                                                  alg.reltol_termination,
                                                  get_terminate_condition(alg,
                                                                          terminate_stats))
  sol = SciMLBase.solve(_prob, alg.alg, args...; callback, kwargs...)

  u, t = if terminate_stats[:best_objective_value_iteration] === nothing
    (sol.u[end], sol.t[end])
  else
    (sol.u[terminate_stats[:best_objective_value_iteration] + 1],
     sol.t[terminate_stats[:best_objective_value_iteration] + 1])
  end

  # Dont count towards NFE since this is mostly a check for convergence
  du = prob.f(u, prob.p, t)

  retcode = (sol.retcode == :Terminated && has_converged(du, u, alg) ? :Success : :Failure)

  return EquilibriumSolution{eltype(uType), ndims(uType), uType, typeof(prob), typeof(alg),
                             typeof(sol.destats), typeof(sol)}(u, du, prob, alg, retcode,
                                                               sol.destats, sol)
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractSteadyStateProblem{uType},
                            alg::DiscreteDEQSolver, args...; maxiters=10,
                            kwargs...) where {uType}
  terminate_stats = Dict{Symbol, Any}(:best_objective_value => real(eltype(prob.u0))(Inf),
                                      :best_objective_value_iteration => nothing)

  us, stats = nlsolve(alg.alg, u -> prob.f(u, prob.p, nothing), prob.u0; maxiters=maxiters,
                      terminate_condition=get_terminate_condition(alg, terminate_stats))

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
                             typeof(destats), typeof(us)}(u, du, prob, alg, retcode,
                                                          destats, us)
end
