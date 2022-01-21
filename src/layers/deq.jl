struct DeepEquilibriumNetwork{J,R,M,P,RE,A,S,K} <: AbstractDeepEquilibriumNetwork
    jacobian_regularization::Bool
    residual_regularization::Bool
    model::M
    p::P
    re::RE
    args::A
    kwargs::K
    sensealg::S
    stats::DEQTrainingStats

    function DeepEquilibriumNetwork(jacobian_regularization, residual_regularization, model, p, re, args, kwargs,
                                    sensealg, stats)
        _p, re = destructure_parameters(model)
        p = p === nothing ? _p : convert(typeof(_p), p)

        return new{jacobian_regularization,residual_regularization,typeof(model),typeof(p),typeof(re),typeof(args),
                   typeof(sensealg),typeof(kwargs)}(jacobian_regularization, residual_regularization, model, p, re,
                                                    args, kwargs, sensealg, stats)
    end
end

Flux.@functor DeepEquilibriumNetwork

function Base.show(io::IO, l::DeepEquilibriumNetwork{J,R}) where {J,R}
    return print(io, "DeepEquilibriumNetwork(jacobian_regularization = $J, residual_regularization = $R) ",
                 string(length(l.p)), " Trainable Parameters")
end

function DeepEquilibriumNetwork(model, solver; jacobian_regularization::Bool=false, residual_regularization::Bool=false,
                                p=nothing, sensealg=get_default_ssadjoint(0.1f0, 0.1f0, 10), kwargs...)
    return DeepEquilibriumNetwork(jacobian_regularization, residual_regularization, model, p, nothing, (solver,),
                                  kwargs, sensealg, DEQTrainingStats(0))
end

function (deq::DeepEquilibriumNetwork)(x::AbstractArray{T}; only_explicit::Bool=false) where {T}
    z = zero(x)
    Zygote.@ignore deq.re(deq.p)(z, x)

    z_star = solve_steady_state_problem(deq.re, deq.p, x, z, deq.sensealg, deq.args...; dudt=nothing,
                                        update_nfe=() -> (deq.stats.nfe += 1), deq.kwargs...)

    jac_loss = (deq.jacobian_regularization ? compute_deq_jacobian_loss(deq.re, deq.p, z_star, x) : T(0))::T

    residual = if deq.residual_regularization
        z_star .- deq.re(deq.p)(z_star, x)
    else
        Zygote.@ignore z_star .- deq.re(deq.p)(z_star, x)
    end

    return z_star, DeepEquilibriumSolution(z_star, z, residual, jac_loss)
end

# For multiple argument functions
function (deq::DeepEquilibriumNetwork)(x::AbstractMatrix{T}, x1::AbstractArray{T}, args...) where {T}
    u0 = zero(x)

    function dudt(u, _p, t)
        deq.stats.nfe += 1
        return first(deq.re(_p)(u, x, x1, args...)) .- u
    end

    ssprob = SteadyStateProblem(dudt, u0, deq.p)
    sol = solve(ssprob, deq.args...; u0=u0, sensealg=deq.sensealg, deq.kwargs...)
    deq.stats.nfe += 1

    z_star = first(deq.re(deq.p)(sol.u, x, x1, args...))

    jac_loss = T(0)

    residual = if deq.residual_regularization
        z_star .- first(deq.re(deq.p)(z_star, x, x1, args...))
    else
        Zygote.@ignore z_star .- first(deq.re(deq.p)(z_star, x, x1, args...))
    end

    return (z_star, x1, args...), DeepEquilibriumSolution(z_star, u0, residual, jac_loss)
end
