"""
    DiscreteDEQSolver(alg=LimitedMemoryBroydenSolver(); mode::Symbol=:rel_deq_default, abstol_termination::T=1.0f-8, reltol_termination::T=1.0f-8)

Solver for Discrete DEQ Problem ([baideep2019](@cite)). Similar to `SSrootfind` but provides more flexibility needed
for solving DEQ problems.

## Arguments

  - `alg`: Algorithm to solve the Nonlinear Problem (Default: [`LimitedMemoryBroydenSolver`](@ref))
  - `mode`: Termination Mode of the solver. See below for a description of the various termination conditions (Default: `:rel_deq_default`)
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

include("discrete/broyden.jl")
include("discrete/limited_memory_broyden.jl")
