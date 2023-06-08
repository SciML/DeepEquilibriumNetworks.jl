abstract type AbstractDeepEquilibriumNetwork <:
              Lux.AbstractExplicitContainerLayer{(:model,)} end

function Lux.initialstates(rng::AbstractRNG, deq::AbstractDeepEquilibriumNetwork)
    rng = Lux.replicate(rng)
    randn(rng, 1)
    return (;
        model=Lux.initialstates(rng, deq.model),
        fixed_depth=Val(0),
        solution=nothing,
        rng)
end

function Lux.initialparameters(rng::AbstractRNG, deq::AbstractDeepEquilibriumNetwork)
    return (; model=Lux.initialparameters(rng, deq.model))
end

abstract type AbstractSkipDeepEquilibriumNetwork <:
              Lux.AbstractExplicitContainerLayer{(:model, :shortcut)} end

function Lux.initialstates(rng::AbstractRNG, deq::AbstractSkipDeepEquilibriumNetwork)
    rng = Lux.replicate(rng)
    randn(rng, 1)
    return (;
        model=Lux.initialstates(rng, deq.model),
        shortcut=Lux.initialstates(rng, deq.shortcut),
        fixed_depth=Val(0),
        solution=nothing,
        rng)
end

const AbstractDEQs = Union{
    AbstractDeepEquilibriumNetwork,
    AbstractSkipDeepEquilibriumNetwork,
}

function (deq::AbstractDEQs)(x::AbstractArray, ps, st::NamedTuple)
    return deq(x, ps, st, _check_unrolled_mode(st))
end

# Utilities
@inline _check_unrolled_mode(::Val{d}) where {d} = Val(d >= 1)
@inline _check_unrolled_mode(st::NamedTuple) = _check_unrolled_mode(st.fixed_depth)

@inline _get_unrolled_depth(::Val{d}) where {d} = d
@inline _get_unrolled_depth(st::NamedTuple) = _get_unrolled_depth(st.fixed_depth)

CRC.@non_differentiable _check_unrolled_mode(::Any)
CRC.@non_differentiable _get_unrolled_depth(::Any)

"""
    DeepEquilibriumSolution(z_star, u₀, residual, jacobian_loss, nfe)

Stores the solution of a DeepEquilibriumNetwork and its variants.

## Fields

  - `z_star`: Steady-State or the value reached due to maxiters
  - `u0`: Initial Condition
  - `residual`: Difference of the ``z^*`` and ``f(z^*, x)``
  - `jacobian_loss`: Jacobian Stabilization Loss (see individual networks to see how it
    can be computed).
  - `nfe`: Number of Function Evaluations
"""
struct DeepEquilibriumSolution{T, R <: AbstractFloat, TRes}
    z_star::T
    u0::T
    residual::TRes
    jacobian_loss::R
    nfe::Int
end
