abstract type AbstractDEQSolver <: AbstractSteadyStateAlgorithm end

"""
    ContinuousDEQSolver(alg=VCAB3(); mode=NLSolveTerminationMode.RelSafeBest,
        abstol=1.0f-8, reltol=1.0f-6, abstol_termination=abstol, reltol_termination=reltol,
        tspan=Inf32, kwargs...)

Solver for Continuous DEQ Problem [pal2022mixing](@cite). Effectively a wrapper around
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
struct ContinuousDEQSolver{A <: DynamicSS} <: AbstractDEQSolver
    alg::A
end

function ContinuousDEQSolver(alg=VCAB3(); mode=NLSolveTerminationMode.RelSafeBest,
        abstol=1.0f-8, reltol=1.0f-6, abstol_termination=abstol, reltol_termination=reltol,
        tspan=Inf32, kwargs...)
    termination_condition = NLSolveTerminationCondition(mode; abstol=abstol_termination,
        reltol=reltol_termination, kwargs...)
    return ContinuousDEQSolver(DynamicSS(alg; abstol, reltol, tspan, termination_condition))
end

"""
    DiscreteDEQSolver(alg = LBroyden(; batched=true,
        termination_condition=NLSolveTerminationCondition(NLSolveTerminationMode.RelSafe;
            abstol=1.0f-8, reltol=1.0f-6))

Solver for Discrete DEQ Problem [baideep2019](@cite). Similar to `SSrootfind` but provides
more flexibility needed for solving DEQ problems.

## Arguments

  - `alg`: Algorithm to solve the Nonlinear Problem. (Default: `LBroyden`)

See also: [`ContinuousDEQSolver`](@ref)
"""
struct DiscreteDEQSolver{A} <: AbstractDEQSolver
    alg::A
    function DiscreteDEQSolver(alg=nothing)
        if alg === nothing
            alg = LBroyden(; batched=true,
                termination_condition=NLSolveTerminationCondition(NLSolveTerminationMode.RelSafe;
                    abstol=1.0f-8, reltol=1.0f-6))
        end
        return new{typeof(alg)}(alg)
    end
end

"""
    EquilibriumSolution

Wraps the solution of a SteadyStateProblem using either ContinuousDEQSolver or
DiscreteDEQSolver. This is mostly an internal implementation detail, which allows proper
dispatch during adjoint computation without type piracy.
"""
@concrete struct EquilibriumSolution{T, N, uType} <: AbstractNonlinearSolution{T, N}
    u::uType
    resid::uType
    prob
    alg
    retcode
end

@truncate_stacktrace EquilibriumSolution 1 2

function SciMLBase.__solve(prob::AbstractSteadyStateProblem, alg::AbstractDEQSolver,
        args...; kwargs...)
    # FIXME: Remove this handle
    sol = solve(prob, alg.alg, args...; kwargs...)

    u, du, retcode = sol.u, sol.resid, sol.retcode

    return EquilibriumSolution{eltype(u), ndims(u)}(u, du, prob, alg, retcode)
end
