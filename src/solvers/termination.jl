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
            return has_converged(DiffEqBase.get_du(integrator), integrator.u, M, abstol, reltol)
        end
        return terminate_condition_closure_2
    end
end

function get_terminate_condition(alg::DiscreteDEQSolver{M,A,T}, args...; kwargs...) where {M,A,T}
    mode = get_mode(M)
    if mode ∈ (:abs_deq_default, :rel_deq_default, :abs_deq_best, :rel_deq_best)
        nstep, protective_threshold, objective_values = 0, T(1e3), T[]

        if mode ∈ (:rel_deq_best, :abs_deq_best)
            @assert length(args) == 1

            args[1][:best_objective_value] = T(Inf)
            args[1][:best_objective_value_iteration] = 0
        end

        function terminate_condition_closure_1(du, u)
            objective = norm(du) / (mode ∈ (:abs_deq_default, :abs_deq_best) ? 1 : (norm(du .+ u) + eps(T)))
            criteria = mode ∈ (:abs_deq_default, :abs_deq_best) ? alg.abstol_termination : alg.reltol_termination

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
        function terminate_condition_closure_2(du, u)
            return has_converged(du, u, M, alg.abstol_termination, alg.reltol_termination)
        end
        return terminate_condition_closure_2
    end
end

# Convergence Criterions
@inline function has_converged(
    du,
    u,
    alg::Union{ContinuousDEQSolver{M},DiscreteDEQSolver{M}},
    abstol=alg.abstol_termination,
    reltol=alg.reltol_termination,
) where {M}
    return has_converged(du, u, M, abstol, reltol)
end

@inline @inbounds function has_converged(du, u, M, abstol, reltol)
    mode = get_mode(M)
    if mode == :norm
        return norm(du) <= abstol && norm(du) <= reltol * norm(du + u)
    elseif mode == :rel
        return all(abs.(du) .<= reltol .* abs.(u))
    elseif mode == :rel_norm
        return norm(du) <= reltol * norm(du + u)
    elseif mode == :rel_deq_default
        return norm(du) <= reltol * norm(du + u)
    elseif mode == :rel_deq_best
        return norm(du) <= reltol * norm(du + u)
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
