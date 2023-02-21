import OrdinaryDiffEq
import SteadyStateDiffEq

"""
    ContinuousDEQSolver(alg=OrdinaryDiffEq.VCABM3(); mode::Symbol=:rel_deq_default,
                        abstol=1f-8, reltol=1f-8, abstol_termination=1f-8,
                        reltol_termination=1f-8, tspan=Inf32)

Solver for Continuous DEQ Problem ([pal2022mixing](@cite)). Similar to `DynamicSS`, but
provides more flexibility needed for solving DEQ problems.

## Arguments

  - `alg`: Algorithm to solve the ODEProblem. (Default: `VCABM3()`)
  - `mode`: Termination Mode of the solver. See below for a description of the various
    termination conditions (Default: `:rel_deq_default`)
  - `abstol`: Absolute tolerance for time stepping. (Default: `1f-8`)
  - `reltol`: Relative tolerance for time stepping. (Default: `1f-8`)
  - `abstol_termination`: Absolute tolerance for termination. (Default: `1f-8`)
  - `reltol_termination`: Relative tolerance for termination. (Default: `1f-8`)
  - `tspan`: Time span. Users should not change this value, instead control termination
    through `maxiters` in `solve` (Default: `Inf32`)

See also: [`DiscreteDEQSolver`](@ref)
"""
struct ContinuousDEQSolver{M, A, T, TS} <: SteadyStateDiffEq.SteadyStateDiffEqAlgorithm
  alg::A
  abstol::T
  reltol::T
  abstol_termination::T
  reltol_termination::T
  tspan::TS
end

function ContinuousDEQSolver(alg=OrdinaryDiffEq.VCABM3(); mode::Symbol=:rel_deq_default,
                             abstol::T=1.0f-8, reltol::T=1.0f-8,
                             abstol_termination::T=1.0f-8, reltol_termination::T=1.0f-8,
                             tspan=Inf32) where {T <: Number}
  return ContinuousDEQSolver{Val(mode), typeof(alg), T, typeof(tspan)}(alg, abstol, reltol,
                                                                       abstol_termination,
                                                                       reltol_termination,
                                                                       tspan)
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
