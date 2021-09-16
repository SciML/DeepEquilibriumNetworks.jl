module FastDEQ

using CUDA
using DiffEqFlux
using DiffEqSensitivity
using Flux
using OrdinaryDiffEq
using SteadyStateDiffEq
using Zygote


struct DeepEquilibriumNetwork{M,P,RE,A,K}
    model::M
    p::P
    re::RE
    args::A
    kwargs::K
end

Flux.@functor DeepEquilibriumNetwork

function DeepEquilibriumNetwork(model, args...; p = nothing, kwargs...)
    _p, re = Flux.destructure(model)
    p = p === nothing ? _p : p
    return DeepEquilibriumNetwork(model, p, re, args, kwargs)
end

Flux.trainable(deq::DeepEquilibriumNetwork) = (deq.p,)

function (deq::DeepEquilibriumNetwork)(x::AbstractArray{T}, p = deq.p) where {T}
    _x = zero(x)
    # Solving the equation f(u) - u = du = 0
    dudt(u, _p, t) = deq.re(_p)(u, x) .- u
    ssprob = SteadyStateProblem(ODEProblem(dudt, _x, (zero(T), one(T)), p))
    return solve(ssprob, deq.args...; u0 = _x, deq.kwargs...).u
end

# Eg:
# model = DeepEquilibriumNetwork(Parallel(+, Dense(1, 1), Dense(1, 1)), DynamicSS(Tsit5(); abstol = 1f-1, reltol = 1f-1))

export DeepEquilibriumNetwork

end
