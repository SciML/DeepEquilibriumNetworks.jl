import ChainRulesCore as CRC
import Lux
import Random
import Statistics

abstract type AbstractDeepEquilibriumNetwork <:
              Lux.AbstractExplicitContainerLayer{(:model,)} end

function Lux.initialstates(rng::Random.AbstractRNG, deq::AbstractDeepEquilibriumNetwork)
  _rng = Lux.replicate(rng)
  randn(_rng, 1)
  return (model=Lux.initialstates(rng, deq.model), fixed_depth=Val(0), solution=nothing,
          rng=_rng)
end

abstract type AbstractSkipDeepEquilibriumNetwork <:
              Lux.AbstractExplicitContainerLayer{(:model, :shortcut)} end

function Lux.initialstates(rng::Random.AbstractRNG, deq::AbstractSkipDeepEquilibriumNetwork)
  _rng = Lux.replicate(rng)
  randn(_rng, 1)
  return (model=Lux.initialstates(rng, deq.model),
          shortcut=Lux.initialstates(rng, deq.shortcut), fixed_depth=Val(0),
          solution=nothing, rng=_rng)
end

@inline _check_unrolled_mode(::Val{d}) where {d} = (d >= 1)::Bool
@inline _check_unrolled_mode(st::NamedTuple)::Bool = _check_unrolled_mode(st.fixed_depth)
@inline _get_unrolled_depth(::Val{d}) where {d} = d::Int
@inline _get_unrolled_depth(st::NamedTuple)::Int = _get_unrolled_depth(st.fixed_depth)

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

## Accessors

We recommend not accessing the fields directly, rather use the functions
`equilibrium_solution`, `initial_condition`, `residual`, `jacobian_loss` and
`number_of_function_evaluations`.
"""
struct DeepEquilibriumSolution{T, R <: AbstractFloat, TRes}
  z_star::T
  u0::T
  residual::TRes
  jacobian_loss::R
  nfe::Int
end

function Base.show(io::IO, l::DeepEquilibriumSolution)
  print(io, "DeepEquilibriumSolution(")
  print(io, "z_star: ", equilibrium_solution(l))
  print(io, ", initial_condition: ", initial_condition(l))
  print(io, ", residual: ", residual(l))
  print(io, ", jacobian_loss: ", jacobian_loss(l))
  print(io, ", NFE: ", number_of_function_evaluations(l))
  print(io, ")")
  return nothing
end

initial_condition(l::DeepEquilibriumSolution) = l.u0
equilibrium_solution(l::DeepEquilibriumSolution) = l.z_star
residual(l::DeepEquilibriumSolution) = l.residual
jacobian_loss(l::DeepEquilibriumSolution) = l.jacobian_loss
number_of_function_evaluations(l::DeepEquilibriumSolution) = l.nfe
function skip_loss(l::DeepEquilibriumSolution)
  return Statistics.mean(abs, equilibrium_solution(l) .- initial_condition(l))
end

function CRC.rrule(::Type{<:DeepEquilibriumSolution}, z_star::T, u0::T, residual::T,
                   jacobian_loss::R, nfe::Int) where {T, R <: AbstractFloat}
  function DeepEquilibriumSolution_pullback(dsol)
    return (CRC.NoTangent(), dsol.z_star, dsol.u0, dsol.residual, dsol.jacobian_loss,
            dsol.nfe)
  end
  return (DeepEquilibriumSolution(z_star, u0, residual, jacobian_loss, nfe),
          DeepEquilibriumSolution_pullback)
end
