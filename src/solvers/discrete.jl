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
function DiscreteDEQSolver(solver=LimitedMemoryBroydenSolver; abstol=1e-8, reltol=1e-8, kwargs...)
    solver = solver(; kwargs..., reltol=reltol, abstol=abstol)
    return SSRootfind(; nlsolve=(f, u0, abstol) -> solver(f, u0))
end

include("discrete/broyden.jl")
include("discrete/limited_memory_broyden.jl")
