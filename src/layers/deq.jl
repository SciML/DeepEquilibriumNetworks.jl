"""
    DeepEquilibriumNetwork(model, solver; jacobian_regularization::Bool=false,
                           sensealg=SteadyStateAdjoint(), kwargs...)

Deep Equilibrium Network as proposed in [baideep2019](@cite).

## Arguments

  - `model`:`Lux` Neural Network.
  - `solver`: Solver for the optimization problem (See: [`ContinuousDEQSolver`](@ref) &
    [`DiscreteDEQSolver`](@ref)).
  - `jacobian_regularization`: If true, Jacobian Loss is computed and stored in the
    [`DeepEquilibriumSolution`](@ref).
  - `sensealg`: See `SciMLSensitivity.SteadyStateAdjoint`.
  - `kwargs`: Additional Parameters that are directly passed to `SciMLBase.solve`.

## Example

```@example
using DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq

model = DeepEquilibriumNetwork(Parallel(+, Dense(2, 2; use_bias=false),
                                        Dense(2, 2; use_bias=false)),
                               ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0);
                               save_everystep=true)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

model(rand(rng, Float32, 2, 1), ps, st)
```

See also: [`SkipDeepEquilibriumNetwork`](@ref), [`MultiScaleDeepEquilibriumNetwork`](@ref),
[`MultiScaleSkipDeepEquilibriumNetwork`](@ref).
"""
struct DeepEquilibriumNetwork{J, M, A, S, K} <: AbstractDeepEquilibriumNetwork
  model::M
  solver::A
  sensealg::S
  kwargs::K
end

@truncate_stacktrace DeepEquilibriumNetwork 1 2

function DeepEquilibriumNetwork(model, solver; jacobian_regularization::Bool=false,
                                sensealg=SteadyStateAdjoint(), kwargs...)
  return DeepEquilibriumNetwork{jacobian_regularization, typeof(model), typeof(solver),
                                typeof(sensealg), typeof(kwargs)}(model, solver, sensealg,
                                                                  kwargs)
end

_jacobian_regularization(::DeepEquilibriumNetwork{J}) where {J} = J

_get_initial_condition(::DeepEquilibriumNetwork, x, ps, st) = zero(x), st

"""
    SkipDeepEquilibriumNetwork(model, shortcut, solver; jacobian_regularization::Bool=false,
                               sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10), kwargs...)

Skip Deep Equilibrium Network as proposed in [pal2022mixing](@cite)

## Arguments

  - `model`: Neural Network.
  - `shortcut`: Shortcut for the network (pass `nothing` for SkipRegDEQ).
  - `solver`: Solver for the optimization problem (See: [`ContinuousDEQSolver`](@ref) &
    [`DiscreteDEQSolver`](@ref)).
  - `jacobian_regularization`: If true, Jacobian Loss is computed and stored in the
    [`DeepEquilibriumSolution`](@ref).
  - `sensealg`: See `SciMLSensitivity.SteadyStateAdjoint`.
  - `kwargs`: Additional Parameters that are directly passed to `SciMLBase.solve`.

## Example

```@example
using DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq

## SkipDEQ
model = SkipDeepEquilibriumNetwork(Parallel(+, Dense(2, 2; use_bias=false),
                                            Dense(2, 2; use_bias=false)), Dense(2, 2),
                                   ContinuousDEQSolver(VCABM3(); abstol=0.01f0,
                                                       reltol=0.01f0); save_everystep=true)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

model(rand(rng, Float32, 2, 1), ps, st)

## SkipRegDEQ
model = SkipDeepEquilibriumNetwork(Parallel(+, Dense(2, 2; use_bias=false),
                                            Dense(2, 2; use_bias=false)), nothing,
                                   ContinuousDEQSolver(VCABM3(); abstol=0.01f0,
                                                       reltol=0.01f0); save_everystep=true)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

model(rand(rng, Float32, 2, 1), ps, st)
```

See also: [`DeepEquilibriumNetwork`](@ref), [`MultiScaleDeepEquilibriumNetwork`](@ref),
[`MultiScaleSkipDeepEquilibriumNetwork`](@ref)
"""
struct SkipDeepEquilibriumNetwork{J, M, Sh, A, S, K} <: AbstractSkipDeepEquilibriumNetwork
  model::M
  shortcut::Sh
  solver::A
  sensealg::S
  kwargs::K
end

@truncate_stacktrace SkipDeepEquilibriumNetwork 1 2 3

function SkipDeepEquilibriumNetwork(model, shortcut, solver; sensealg=SteadyStateAdjoint(),
                                    jacobian_regularization::Bool=false, kwargs...)
  return SkipDeepEquilibriumNetwork{jacobian_regularization, typeof(model),
                                    typeof(shortcut), typeof(solver), typeof(sensealg),
                                    typeof(kwargs)}(model, shortcut, solver, sensealg,
                                                    kwargs)
end

_jacobian_regularization(::SkipDeepEquilibriumNetwork{J}) where {J} = J

function _get_initial_condition(deq::SkipDeepEquilibriumNetwork{J, M, Nothing}, x, ps,
                                st) where {J, M}
  z, st_ = deq.model((zero(x), x), ps.model, st.model)
  @set! st.model = st_
  return z, st
end

function _get_initial_condition(deq::SkipDeepEquilibriumNetwork, x, ps, st)
  z, st_ = deq.shortcut(x, ps.shortcut, st.shortcut)
  @set! st.shortcut = st_
  return z, st
end
