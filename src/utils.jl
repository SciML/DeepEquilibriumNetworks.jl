function _init_identity_matrix(x::AbstractArray{T}, scale::T = T(1)) where {T}
    x_ = vec(x)
    return _init_identity_matrix!(x_ .* x_', scale)
end

function _init_identity_matrix!(x::AbstractMatrix{T}, scale::T = T(1)) where {T}
    x .= zero(T)
    idxs = diagind(x)
    @. @view(x[idxs]) = scale * true
    return x
end

function generate_model_trajectory(deq::AbstractDeepEquilibriumModel, x, max_depth::Int,
                                   abstol::T = 1e-8, reltol::T = 1e-8) where {T}
    deq_func = construct_iterator(deq, x)
    values = [x, deq_func()]
    for i = 2:max_depth
        sol = deq_func()
        push!(values, sol)
        if (norm(sol .- values[end - 1]) ≤ abstol) || (norm(sol .- values[end - 1]) / norm(values[end - 1]) ≤ reltol)
            return values
        end
    end
    return values
end