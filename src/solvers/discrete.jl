# Wrapper for Discrete DEQs
"""
    DiscreteDEQSolver(solver=LimitedMemoryBroydenSolver; abstol=1e-8, reltol=1e-8, kwargs...)

Solver for Discrete DEQ Problem ([baideep2019](@cite)). A wrapper around `SSrootfind` to mimic the [`ContinuousDEQSolver`](@ref) API. 

## Arguments

* `solver`: NonLinear Solver for the DEQ problem. (Default: [`LimitedMemoryBroydenSolver`](@ref))
* `abstol`: Absolute tolerance for termination. (Default: `1e-8`)
* `reltol`: Relative tolerance for termination. (Default: `1e-8`)
* `kwargs`: Additional keyword arguments passed to the solver.

!!! note
    There is no `mode` kwarg for [`DiscreteDEQSolver`](@ref). Instead solvers directly define their own termination condition.
    For [`BroydenSolver`](@ref) and [`LimitedMemoryBroydenSolver`](@ref), the termination conditions are `:abs_norm` &
    `:rel_deq_default` respectively.

See also: [`ContinuousDEQSolver`](@ref)
"""
struct DiscreteDEQSolver{M,A,T} <: SteadyStateDiffEq.SteadyStateDiffEqAlgorithm
    alg::A
    abstol_termination::T
    reltol_termination::T
end

function DiscreteDEQSolver(
    alg=LimitedMemoryBroydenSolver();
    mode::Symbol=:rel_deq_default,
    abstol_termination::T=1.0f-8,
    reltol_termination::T=1.0f-8
) where {T<:Number}
    return DiscreteDEQSolver{Val(mode),typeof(alg),T}(alg, abstol_termination, reltol_termination)
end

include("discrete/broyden.jl")
include("discrete/limited_memory_broyden.jl")
