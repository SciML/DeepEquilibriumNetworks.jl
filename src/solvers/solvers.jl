# NOTE(@avik-pal): Keeping the ContinuousDEQSolver struct for now, as I feel the defaults
#                  for typical SteadyStateProblems are not really appropriate for DEQs.
"""
    ContinuousDEQSolver(alg=VCABM3(); mode=SteadyStateTerminationMode.RelSafeBest,
                        abstol=1.0f-8, reltol=1.0f-6, abstol_termination=abstol,
                        reltol_termination=reltol, tspan=Inf32, kwargs...)

Solver for Continuous DEQ Problem ([pal2022mixing](@cite)). Effectively a wrapper around
`DynamicSS` with more sensible defaults for DEQs.

## Arguments

  - `alg`: Algorithm to solve the ODEProblem. (Default: `VCAB3()`)
  - `mode`: Termination Mode of the solver. See the documentation for
    `NLSolveTerminationCondition` for more information.
    (Default: `NLSolveTerminationMode.RelSafeBest`)
  - `abstol`: Absolute tolerance for time stepping. (Default: `1f-8`)
  - `reltol`: Relative tolerance for time stepping. (Default: `1f-6`)
  - `abstol_termination`: Absolute tolerance for termination. (Default: `abstol`)
  - `reltol_termination`: Relative tolerance for termination. (Default: `reltol`)
  - `tspan`: Time span. Users should not change this value, instead control termination
    through `maxiters` in `solve`. (Default: `Inf32`)
  - `kwargs`: Additional Parameters that are directly passed to
    `NLSolveTerminationCondition`.

See also: [`DiscreteDEQSolver`](@ref)
"""
struct ContinuousDEQSolver{A <: DynamicSS} <: SteadyStateDiffEqAlgorithm
  alg::A
end

function ContinuousDEQSolver(alg=VCAB3(); mode=NLSolveTerminationMode.RelSafeBest,
                             abstol=1.0f-8, reltol=1.0f-6, abstol_termination=abstol,
                             reltol_termination=reltol, tspan=Inf32, kwargs...)
  termination_condition = NLSolveTerminationCondition(mode; abstol=abstol_termination,
                                                      reltol=reltol_termination, kwargs...)
  return ContinuousDEQSolver(DynamicSS(alg; abstol, reltol, tspan, termination_condition))
end

"""
    DiscreteDEQSolver(alg=LimitedMemoryBroydenSolver(); mode::Symbol=:rel_deq_default,
                      abstol_termination::T=1.0f-8, reltol_termination::T=1.0f-8)

Solver for Discrete DEQ Problem ([baideep2019](@cite)). Similar to `SSrootfind`, but provides
more flexibility needed for solving DEQ problems.

## Arguments

  - `alg`: Algorithm to solve the Nonlinear Problem.
    (Default: [`LimitedMemoryBroydenSolver`](@ref))
  - `mode`: Termination Mode of the solver. See the documentation for
    `SteadyStateTerminationCriteria` for more information.
    (Default: `NLSolveTerminationMode.RelSafe`)
  - `abstol_termination`: Absolute tolerance for termination. (Default: `1f-8`)
  - `reltol_termination`: Relative tolerance for termination. (Default: `1f-6`)

See also: [`ContinuousDEQSolver`](@ref)
"""
struct DiscreteDEQSolver{A, TC} <: SteadyStateDiffEqAlgorithm
  alg::A
  termination_condition::TC
end

function DiscreteDEQSolver(alg=LimitedMemoryBroydenSolver();
                           mode=NLSolveTerminationMode.RelSafe, abstol_termination=1.0f-8,
                           reltol_termination=1.0f-6, kwargs...)
  termination_condition = NLSolveTerminationCriteria(mode; abstol=abstol_termination,
                                                     reltol=reltol_termination, kwargs...)
  # TODO: Get termination criterias into NonlinearSolve.jl or SimpleNonlinearSolve.jl
  return DiscreteDEQSolver(alg, termination_condition)
end
