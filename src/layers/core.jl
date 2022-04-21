abstract type AbstractDeepEquilibriumNetwork <: AbstractExplicitContainerLayer{(:model,)} end

function initialstates(rng::AbstractRNG, deq::AbstractDeepEquilibriumNetwork)
    return (model=initialstates(rng, deq.model), fixed_depth=0)
end

abstract type AbstractSkipDeepEquilibriumNetwork <: AbstractExplicitContainerLayer{(:model,:shortcut)} end

function initialstates(rng::AbstractRNG, deq::AbstractSkipDeepEquilibriumNetwork)
    return (
        model=initialstates(rng, deq.model), shortcut=initialstates(rng, deq.shortcut), fixed_depth=0
    )
end

"""
    DeepEquilibriumSolution(z_star, u₀, residual, jacobian_loss, nfe)

Stores the solution of a DeepEquilibriumNetwork and its variants.

## Fields
    * `z_star`: Steady-State or the value reached due to maxiters
    * `u₀`: Initial Condition
    * `residual`: Difference of the ``z^*`` and ``f(z^*, x)``
    * `jacobian_loss`: Jacobian Stabilization Loss (see individual networks to see how it can be computed)
    * `nfe`: Number of Function Evaluations
"""
struct DeepEquilibriumSolution{T,R<:AbstractFloat}
    z_star::T
    u₀::T
    residual::T
    jacobian_loss::R
    nfe::Int
end

function Base.show(io::IO, l::DeepEquilibriumSolution)
    print(io, "DeepEquilibriumSolution(")
    print(io, "z_star: ", l.z_star)
    print(io, ", initial_condition: ", l.u₀)
    print(io, ", residual: ", l.residual)
    print(io, ", jacobian_loss: ", l.jacobian_loss)
    print(io, ", NFE: ", l.nfe)
    print(io, ")")
    return nothing
end