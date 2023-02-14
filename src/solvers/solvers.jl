# NOTE(@avik-pal): Keeping the ContinuousDEQSolver struct for now, as I feel the defaults
#                  for typical SteadyStateProblems are not really appropriate for DEQs.
#                  Also allows me to create a custom solution which I will remove in later
#                  steps.
"""
    ContinuousDEQSolver(alg=VCABM3(); mode=SteadyStateTerminationMode.RelSafeBest,
                        abstol=1.0f-8, reltol=1.0f-6, abstol_termination=1.0f-8,
                        reltol_termination=1.0f-6, tspan=Inf32, kwargs...)

Solver for Continuous DEQ Problem ([pal2022mixing](@cite)). Effectively a wrapper around
`DynamicSS` with more sensible defaults for DEQs.

## Arguments

  - `alg`: Algorithm to solve the ODEProblem. (Default: `VCAB3()`)
  - `mode`: Termination Mode of the solver. See the documentation for
    `SteadyStateTerminationCriteria` for more information.
    (Default: `SteadyStateTerminationMode.RelSafeBest`)
  - `abstol`: Absolute tolerance for time stepping. (Default: `1f-8`)
  - `reltol`: Relative tolerance for time stepping. (Default: `1f-6`)
  - `abstol_termination`: Absolute tolerance for termination. (Default: `abstol`)
  - `reltol_termination`: Relative tolerance for termination. (Default: `reltol`)
  - `tspan`: Time span. Users should not change this value, instead control termination
    through `maxiters` in `solve` (Default: `Inf32`)
  - `kwargs`: Additional Parameters that are directly passed to
    `SteadyStateTerminationCriteria`.

See also: [`DiscreteDEQSolver`](@ref)
"""
struct ContinuousDEQSolver{A <: DynamicSS} <: SteadyStateDiffEqAlgorithm
  alg::A
end

function ContinuousDEQSolver(alg=VCAB3(); mode=SteadyStateTerminationMode.RelSafeBest,
                             abstol=1.0f-8, reltol=1.0f-6, abstol_termination=abstol,
                             reltol_termination=reltol, tspan=Inf32, kwargs...)
  termination_condition = SteadyStateTerminationCriteria(mode; abstol=abstol_termination,
                                                         reltol=reltol_termination,
                                                         kwargs...)
  ss_alg = DynamicSS(alg; abstol, reltol, tspan, termination_condition)
  return ContinuousDEQSolver{typeof(ss_alg)}(ss_alg)
end

"""
    DiscreteDEQSolver(alg=LimitedMemoryBroydenSolver(); mode::Symbol=:rel_deq_default,
                      abstol_termination::T=1.0f-8, reltol_termination::T=1.0f-8)

Solver for Discrete DEQ Problem ([baideep2019](@cite)). Similar to `SSrootfind`, but provides
more flexibility needed for solving DEQ problems.

## Arguments

  - `alg`: Algorithm to solve the Nonlinear Problem.
    (Default: [`LimitedMemoryBroydenSolver`](@ref))
  - `mode`: Termination Mode of the solver. See below for a description of the various
    termination conditions. (Default: `:rel_deq_default`)
  - `abstol_termination`: Absolute tolerance for termination. (Default: `1f-8`)
  - `reltol_termination`: Relative tolerance for termination. (Default: `1f-8`)

See also: [`ContinuousDEQSolver`](@ref)
"""
struct DiscreteDEQSolver{M, A, T} <: SteadyStateDiffEq.SteadyStateDiffEqAlgorithm
  alg::A
  abstol_termination::T
  reltol_termination::T
end

function DiscreteDEQSolver(alg=LimitedMemoryBroydenSolver(); mode::Symbol=:rel_deq_default,
                           abstol_termination::T=1.0f-8,
                           reltol_termination::T=1.0f-8) where {T <: Number}
  return DiscreteDEQSolver{Val(mode), typeof(alg), T}(alg, abstol_termination,
                                                      reltol_termination)
end
