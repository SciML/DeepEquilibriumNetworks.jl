function solve_steady_state_problem(re, p, x, u0, sensealg, args...; dudt=nothing, update_nfe=() -> (), kwargs...)
    # Solving the equation f(u) - u = du = 0
    update_is_variational_hidden_dropout_mask_reset_allowed(false)

    dudt_ = if dudt === nothing
        function (u, _p, t)
            update_nfe()
            return re(_p)(u, x) .- u
        end
    else
        dudt
    end

    ssprob = SteadyStateProblem(dudt_, u0, p)
    sol = solve(ssprob, args...; u0=u0, sensealg=sensealg, kwargs...)

    z = re(p)(sol.u, x)::typeof(x)
    update_nfe()

    update_is_variational_hidden_dropout_mask_reset_allowed(true)

    return z
end

function solve_depth_k_neural_network(re, p, x, u0, depth)
    update_is_variational_hidden_dropout_mask_reset_allowed(false)
    model = re(p)
    for _ in 1:depth
        u0 = model(u0, x)
    end
    update_is_variational_hidden_dropout_mask_reset_allowed(true)
    return u0
end


flatten(x::AbstractArray{T,N}) where {T,N} = reshape(x, :, size(x, N))

Zygote.@adjoint function flatten(x::AbstractArray{T,N}) where {T,N}
    s = size(x)
    res = reshape(x, :, s[N])
    flatten_sensitivity(Δ) = (reshape(Δ, s),)
    return res, flatten_sensitivity
end


struct NoOp end

(noop::NoOp)(x...) = x
(noop::NoOp)(x) = x
