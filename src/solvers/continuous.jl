"""
    ContinuousDEQSolver(alg=VCABM4(); mode::Symbol=:rel_deq_default, abstol=1f-8, reltol=1f-8, abstol_termination=1f-8, reltol_termination=1f-8, tspan=Inf32)

Solver for Continuous DEQ Problem ([pal2022mixing](@cite)). Similar to `DynamicSS` but provides more flexibility needed
for solving DEQ problems.

## Arguments

* `alg`: Algorithm to solve the ODEProblem. (Default: `VCABM4()`)
* `mode`: Termination Mode of the solver. See below for a description of the various termination conditions (Default: `:rel_deq_default`)
* `abstol`: Absolute tolerance for time stepping. (Default: `1f-8`)
* `reltol`: Relative tolerance for time stepping. (Default: `1f-8`)
* `abstol_termination`: Absolute tolerance for termination. (Default: `1f-8`)
* `reltol_termination`: Relative tolerance for termination. (Default: `1f-8`)
* `tspan`: Time span. Users should not change this value, instead control termination through `maxiters` in `solve` (Default: `Inf32`)

## Termination Modes

#### Termination on Absolute Tolerance

* `:abs`: Terminates if ``all \\left( | \\frac{\\partial u}{\\partial t} | \\leq abstol \\right)``
* `:abs_norm`: Terminates if ``\\| \\frac{\\partial u}{\\partial t} \\| \\leq abstol``
* `:abs_deq_default`: Essentially `abs_norm` + terminate if there has been no improvement for the last 30 steps + terminate if the solution blows up (diverges)
* `:abs_deq_best`: Same as `:abs_deq_default` but uses the best solution found so far, i.e. deviates only if the solution has not converged

#### Termination on Relative Tolerance

* `:rel`: Terminates if ``all \\left(| \\frac{\\partial u}{\\partial t} | \\leq reltol \\times | u | \\right)``
* `:rel_norm`: Terminates if ``\\| \\frac{\\partial u}{\\partial t} \\| \\leq reltol \\times \\| \\frac{\\partial u}{\\partial t} + u \\|``
* `:rel_deq_default`: Essentially `rel_norm` + terminate if there has been no improvement for the last 30 steps + terminate if the solution blows up (diverges)
* `:rel_deq_best`: Same as `:rel_deq_default` but uses the best solution found so far, i.e. deviates only if the solution has not converged

#### Termination using both Absolute and Relative Tolerances

* `:norm`: Terminates if ``\\| \\frac{\\partial u}{\\partial t} \\| \\leq reltol \\times \\| \\frac{\\partial u}{\\partial t} + u \\|`` &
           ``\\| \\frac{\\partial u}{\\partial t} \\| \\leq abstol``
* `fallback`: Check if all values of the derivative is close to zero wrt both relative and absolute tolerance. This is usable for small problems
              but doesn't scale well for neural networks, and should be avoided unless absolutely necessary

See also: [`DiscreteDEQSolver`](@ref)

!!! note 
    This  will be upstreamed to DiffEqSensitivity in the later releases of the package
"""
struct ContinuousDEQSolver{M,A,T,TS} <: SteadyStateDiffEq.SteadyStateDiffEqAlgorithm
    alg::A
    abstol::T
    reltol::T
    abstol_termination::T
    reltol_termination::T
    tspan::TS
end

function ContinuousDEQSolver(
    alg=VCABM3();
    mode::Symbol=:rel_deq_default,
    abstol::T=1.0f-8,
    reltol::T=1.0f-8,
    abstol_termination::T=1.0f-8,
    reltol_termination::T=1.0f-8,
    tspan=Inf32,
) where {T<:Number}
    return ContinuousDEQSolver{Val(mode),typeof(alg),T,typeof(tspan)}(
        alg, abstol, reltol, abstol_termination, reltol_termination, tspan
    )
end

get_mode(::Val{mode}) where {mode} = mode

function get_terminate_condition(alg::ContinuousDEQSolver{M,A,T}, args...; kwargs...) where {M,A,T}
    mode = get_mode(M)
    if mode ∈ (:abs_deq_default, :rel_deq_default, :abs_deq_best, :rel_deq_best)
        nstep, protective_threshold, objective_values = 0, T(1e3), T[]

        if mode ∈ (:rel_deq_best, :abs_deq_best)
            @assert length(args) == 1

            args[1][:best_objective_value] = T(Inf)
            args[1][:best_objective_value_iteration] = 0
        end

        function terminate_condition_closure_1(integrator, abstol, reltol, min_t)
            du, u = DiffEqBase.get_du(integrator), integrator.u
            objective = norm(du) / (mode ∈ (:abs_deq_default, :abs_deq_best) ? 1 : (norm(du .+ u) + eps(T)))
            criteria = mode ∈ (:abs_deq_default, :abs_deq_best) ? abstol : reltol

            if mode ∈ (:rel_deq_best, :abs_deq_best)
                if objective < args[1][:best_objective_value]
                    args[1][:best_objective_value] = objective
                    args[1][:best_objective_value_iteration] = nstep + 1
                end
            end

            # Main Termination Criteria
            objective <= criteria && return true

            # Terminate if there has been no improvement for the last 30 steps
            nstep += 1
            push!(objective_values, objective)

            objective <= 3 * criteria &&
                nstep >= 30 &&
                maximum(objective_values[max(1, length(objective_values) - nstep):end]) <
                1.3 * minimum(objective_values[max(1, length(objective_values) - nstep):end]) &&
                return true

            # Protective break
            objective >= objective_values[1] * protective_threshold * length(du) && return true

            return false
        end
        return terminate_condition_closure_1
    else
        function terminate_condition_closure_2(integrator, abstol, reltol, min_t)
            return has_converged(DiffEqBase.get_du(integrator), integrator.u, alg, abstol, reltol)
        end
        return terminate_condition_closure_2
    end
end

# Convergence Criterions
function has_converged(
    du, u, alg::ContinuousDEQSolver{M}, abstol=alg.abstol_termination, reltol=alg.reltol_termination
) where {M}
    mode = get_mode(M)
    if mode == :norm
        return norm(du) <= abstol && norm(du) <= reltol * norm(du .+ u)
    elseif mode == :rel
        return all(abs.(du) .<= reltol .* abs.(u))
    elseif mode == :rel_norm
        return norm(du) <= reltol * norm(du .+ u)
    elseif mode == :rel_deq_default
        return norm(du) <= reltol * norm(du .+ u)
    elseif mode == :rel_deq_best
        return norm(du) <= reltol * norm(du .+ u)
    elseif mode == :abs
        return all(abs.(du) .<= abstol)
    elseif mode == :abs_norm
        return norm(du) <= abstol
    elseif mode == :abs_deq_default
        return norm(du) <= abstol
    elseif mode == :abs_deq_best
        return norm(du) <= abstol
    else
        return all(abs.(du) .<= abstol .& abs.(du) .<= reltol .* abs.(u))
    end
end
