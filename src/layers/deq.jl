@generated function _evaluate_unrolled_deq(model, z_star, x, ps, st, ::Val{d}) where {d}
  calls = [:((z_star, st) = model((z_star, x), ps, st)) for _ in 1:d]
  push!(calls, :(return z_star, st))
  return Expr(:block, calls...)
end

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

# Main Functions
const SingleScaleDeepEquilibriumNetworks = Union{SkipDeepEquilibriumNetwork,
                                                 DeepEquilibriumNetwork}

function (deq::SingleScaleDeepEquilibriumNetworks)(x::AbstractArray{T}, ps, st::NamedTuple,
                                                   ::Val{true}) where {T}
  # Pretraining without Fixed Point Solving
  z, st = _get_initial_condition(deq, x, ps, st)
  depth = _get_unrolled_depth(st)

  z_star, st_ = _evaluate_unrolled_deq(deq.model, z, x, ps.model, st.model, st.fixed_depth)

  residual = CRC.ignore_derivatives(z_star .- deq.model((z_star, x), ps.model, st.model)[1])

  @set! st.model = st_
  @set! st.solution = DeepEquilibriumSolution(z_star, z, residual, T(0), depth)

  return z_star, st
end

function (deq::SingleScaleDeepEquilibriumNetworks)(x::AbstractArray{T}, ps, st::NamedTuple,
                                                   ::Val{false}) where {T}
  z, st = _get_initial_condition(deq, x, ps, st)
  st_, nfe = st.model, 0

  function dudt(u, p, t)
    nfe += 1
    u_, st_ = deq.model((u, x), p, st_)
    return u_ .- u
  end

  prob = SteadyStateProblem(ODEFunction{false}(dudt), z, ps.model)
  sol = solve(prob, deq.solver; deq.sensealg, deq.kwargs...)

  z_star, st_ = deq.model((sol.u, x), ps.model, st_)

  # if _jacobian_regularization(deq)
  #   rng = Lux.replicate(st.rng)
  #   jac_loss = estimate_jacobian_trace(Val(:finite_diff), deq.model, ps, st.model, z_star,
  #                                      x, rng)
  # else
  #   rng = st.rng
  #   jac_loss = T(0)
  # end
  jac_loss = T(0)

  residual = CRC.ignore_derivatives(z_star .- deq.model((z_star, x), ps.model, st.model)[1])

  @set! st.model = st_
  # @set! st.rng = rng
  @set! st.solution = DeepEquilibriumSolution(z_star, z, residual, jac_loss, nfe)

  return z_star, st
end
