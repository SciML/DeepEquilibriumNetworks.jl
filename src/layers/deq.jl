struct DeepEquilibriumNetwork{J,R,M,P,RE,A,S,K} <: AbstractDeepEquilibriumNetwork
    jacobian_regularization::Bool
    return_residual_error::Bool
    model::M
    p::P
    re::RE
    args::A
    kwargs::K
    sensealg::S
    stats::DEQTrainingStats

    function DeepEquilibriumNetwork(jacobian_regularization, return_residual_error, model, p, re, args, kwargs, sensealg, stats)
        _p, re = destructure_parameters(model)
        p = p === nothing ? _p : convert(typeof(_p), p)

        return new{jacobian_regularization,return_residual_error,typeof(model),typeof(p),typeof(re),typeof(args),typeof(sensealg),
                   typeof(kwargs)}(jacobian_regularization, return_residual_error, model, p, re, args, kwargs, sensealg, stats)
    end
end

Flux.@functor DeepEquilibriumNetwork

function Base.show(io::IO, l::DeepEquilibriumNetwork{J,R}) where {J,R}
    return print(io, "DeepEquilibriumNetwork(jacobian_regularization = $J, residual_regularization = $R) ", string(length(l.p)), " Trainable Parameters")
end

function DeepEquilibriumNetwork(model, solver; jacobian_regularization::Bool=false, return_residual_error::Bool=false, p=nothing,
                                sensealg=get_default_ssadjoint(0.1f0, 0.1f0, 10), kwargs...)
    return DeepEquilibriumNetwork(jacobian_regularization, return_residual_error, model, p, nothing, (solver,), kwargs, sensealg,
                                  DEQTrainingStats(0))
end

function (deq::DeepEquilibriumNetwork{false,false})(x::AbstractArray{T}) where {T}
    # Solving the equation f(u) - u = du = 0
    z = zero(x)
    Zygote.@ignore deq.re(deq.p)(z, x)

    return solve_steady_state_problem(deq.re, deq.p, x, z, deq.sensealg, deq.args...; dudt=nothing,
                                      update_nfe=() -> (deq.stats.nfe += 1), deq.kwargs...)
end

function (deq::DeepEquilibriumNetwork{false,true})(x::AbstractArray{T}) where {T}
    # Solving the equation f(u) - u = du = 0
    z = zero(x)
    Zygote.@ignore deq.re(deq.p)(z, x)

    z_star = solve_steady_state_problem(deq.re, deq.p, x, z, deq.sensealg, deq.args...; dudt=nothing,
                                        update_nfe=() -> (deq.stats.nfe += 1), deq.kwargs...)

    return z_star, sum(abs, z_star .- deq.re(deq.p)(z_star, x))
end

function (deq::DeepEquilibriumNetwork{true,false})(x::AbstractArray{T}) where {T}
    # Solving the equation f(u) - u = du = 0
    z = zero(x)
    Zygote.@ignore deq.re(deq.p)(z, x)

    z_star = solve_steady_state_problem(deq.re, deq.p, x, z, deq.sensealg, deq.args...; dudt=nothing,
                                        update_nfe=() -> (deq.stats.nfe += 1), deq.kwargs...)

    jac_loss = compute_deq_jacobian_loss(deq.re, deq.p, z_star, x)

    return z_star, jac_loss
end

function (deq::DeepEquilibriumNetwork{true,true})(x::AbstractArray{T}) where {T}
    # Solving the equation f(u) - u = du = 0
    z = zero(x)
    Zygote.@ignore deq.re(deq.p)(z, x)

    z_star = solve_steady_state_problem(deq.re, deq.p, x, z, deq.sensealg, deq.args...; dudt=nothing,
                                        update_nfe=() -> (deq.stats.nfe += 1), deq.kwargs...)

    jac_loss = compute_deq_jacobian_loss(deq.re, deq.p, z_star, x)

    return z_star, jac_loss, sum(abs, z_star .- deq.re(deq.p)(z_star, x))
end

function (deq::DeepEquilibriumNetwork{false})(lapl::AbstractMatrix, x::AbstractMatrix)
    # NOTE: encoded_features being 0 causes NaN gradients to propagate
    u0 = deq.re(p)(lapl, zero(x) .+ eps(eltype(x)), x)[2]
    deq.stats.nfe += 1

    function dudt(u, _p, t)
        deq.stats.nfe += 1
        return deq.re(_p)(lapl, u, x)[2] .- u
    end

    ssprob = SteadyStateProblem(dudt, u0, p)
    sol = solve(ssprob, deq.args...; u0=u0, sensealg=deq.sensealg, deq.kwargs...)
    deq.stats.nfe += 1

    return deq.re(p)(lapl, sol.u, x)
end
