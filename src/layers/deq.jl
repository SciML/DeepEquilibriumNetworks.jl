import ChainRulesCore as CRC
import SteadyStateDiffEq
import OrdinaryDiffEq

"""
    DeepEquilibriumNetwork(model, solver; jacobian_regularization::Bool=false,
                           sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10), kwargs...)

Deep Equilibrium Network as proposed in [baideep2019](@cite).

## Arguments

  - `model`:`Lux` Neural Network.
  - `solver`: Solver for the optimization problem (See: [`ContinuousDEQSolver`](@ref) &
    [`DiscreteDEQSolver`](@ref)).
  - `jacobian_regularization`: If true, Jacobian Loss is computed and stored in the
    [`DeepEquilibriumSolution`](@ref).
  - `sensealg`: See [`DeepEquilibriumAdjoint`](@ref).
  - `kwargs`: Additional Parameters that are directly passed to `SciMLBase.solve`.

## Example

```julia
import DeepEquilibriumNetworks as DEQs
import Lux
import Random
import OrdinaryDiffEq

model = DEQs.DeepEquilibriumNetwork(Lux.Parallel(+, Lux.Dense(2, 2; bias=false),
                                                 Lux.Dense(2, 2; bias=false)),
                                    DEQs.ContinuousDEQSolver(OrdinaryDiffEq.VCABM3();
                                                             abstol=0.01f0, reltol=0.01f0))

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

function DeepEquilibriumNetwork(model, solver; jacobian_regularization::Bool=false,
                                sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10),
                                kwargs...)
  return DeepEquilibriumNetwork{jacobian_regularization, typeof(model), typeof(solver),
                                typeof(sensealg), typeof(kwargs)}(model, solver, sensealg,
                                                                  kwargs)
end

function (deq::DeepEquilibriumNetwork{J})(x::AbstractArray{T}, ps,
                                          st::NamedTuple) where {J, T}
  z = zero(x)

  if _check_unrolled_mode(st)
    # Pretraining without Fixed Point Solving
    st_ = st.model
    z_star = z

    for _ in 1:_get_unrolled_depth(st)
      z_star, st_ = deq.model((z_star, x), ps, st_)
    end

    residual = CRC.ignore_derivatives(z_star .- deq.model((z_star, x), ps, st.model)[1])
    sol = UnrolledDEQSolution(z_star, residual, (; nf=_get_unrolled_depth(st)))
    st = merge(st,
               (model=st_,
                solution=DeepEquilibriumSolution(z_star, z, residual, 0.0f0,
                                                 _get_unrolled_depth(st), sol)))

    return z_star, st
  end

  st_ = st.model

  function dudt(u, p, t)
    u_, st_ = deq.model((u, x), p, st_)
    return u_ .- u
  end

  prob = SteadyStateDiffEq.SteadyStateProblem(OrdinaryDiffEq.ODEFunction{false}(dudt), z,
                                              ps)
  sol = SciMLBase.solve(prob, deq.solver; sensealg=deq.sensealg, deq.kwargs...)
  z_star, st_ = deq.model((sol.u, x), ps, st.model)

  if J
    rng = Lux.replicate(st.rng)
    jac_loss = estimate_jacobian_trace(Val(:finite_diff), deq.model, ps, st.model, z_star,
                                       x, rng)
  else
    rng = st.rng
    jac_loss = T(0)
  end
  residual = CRC.ignore_derivatives(z_star .- deq.model((z_star, x), ps, st.model)[1])

  st = merge(st,
             (model=st_, rng=rng,
              solution=DeepEquilibriumSolution(z_star, z, residual, jac_loss,
                                               sol.destats.nf + 1 + J, sol)))

  return z_star, st
end

"""
    SkipDeepEquilibriumNetwork(model, shortcut, solver; jacobian_regularization::Bool=false,
                               sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10), kwargs...)

Skip Deep Equilibrium Network as proposed in [pal2022mixing](@cite)

## Arguments

  - `model`: Neural Network.
  - `shortcut`: Shortcut for the network (pass `nothing` for SkipDEQV2).
  - `solver`: Solver for the optimization problem (See: [`ContinuousDEQSolver`](@ref) &
    [`DiscreteDEQSolver`](@ref)).
  - `jacobian_regularization`: If true, Jacobian Loss is computed and stored in the
    [`DeepEquilibriumSolution`](@ref).
  - `sensealg`: See [`DeepEquilibriumAdjoint`](@ref).
  - `kwargs`: Additional Parameters that are directly passed to `SciMLBase.solve`.

## Example

```julia
import DeepEquilibriumNetworks as DEQs
import Lux
import Random
import OrdinaryDiffEq

## SkipDEQ
model = DEQs.SkipDeepEquilibriumNetwork(Lux.Parallel(+, Lux.Dense(2, 2; bias=false),
                                                     Lux.Dense(2, 2; bias=false)),
                                        Lux.Dense(2, 2),
                                        DEQs.ContinuousDEQSolver(OrdinaryDiffEq.VCABM3();
                                                                 abstol=0.01f0,
                                                                 reltol=0.01f0))

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

model(rand(rng, Float32, 2, 1), ps, st)

## SkipDEQV2
model = DEQs.SkipDeepEquilibriumNetwork(Lux.Parallel(+, Lux.Dense(2, 2; bias=false),
                                                     Lux.Dense(2, 2; bias=false)), nothing,
                                        DEQs.ContinuousDEQSolver(OrdinaryDiffEq.VCABM3();
                                                                 abstol=0.01f0,
                                                                 reltol=0.01f0))

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

function SkipDeepEquilibriumNetwork(model, shortcut, solver;
                                    jacobian_regularization::Bool=false,
                                    sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10),
                                    kwargs...)
  return SkipDeepEquilibriumNetwork{jacobian_regularization, typeof(model),
                                    typeof(shortcut), typeof(solver), typeof(sensealg),
                                    typeof(kwargs)}(model, shortcut, solver, sensealg,
                                                    kwargs)
end

function (deq::SkipDeepEquilibriumNetwork{J, M, S})(x::AbstractArray{T}, ps,
                                                    st::NamedTuple) where {J, M, S, T}
  z, st = if S == Nothing
    z__, st__ = deq.model((zero(x), x), ps.model, st.model)
    z__, merge(st, (model=st__,))
  else
    z__, st__ = deq.shortcut(x, ps.shortcut, st.shortcut)
    z__, merge(st, (shortcut=st__,))
  end

  if _check_unrolled_mode(st)
    # Pretraining without Fixed Point Solving
    st_ = st.model
    z_star = z
    for _ in 1:_get_unrolled_depth(st)
      z_star, st_ = deq.model((z_star, x), ps.model, st_)
    end

    residual = CRC.ignore_derivatives(z_star .-
                                      deq.model((z_star, x), ps.model, st.model)[1])
    sol = UnrolledDEQSolution(z_star, residual, (; nf=_get_unrolled_depth(st)))
    st = merge(st,
               (model=st_,
                solution=DeepEquilibriumSolution(z_star, z, residual, 0.0f0,
                                                 _get_unrolled_depth(st), sol)))

    return z_star, st
  end

  st_ = st.model

  function dudt(u, p, t)
    u_, st_ = deq.model((u, x), p, st_)
    return u_ .- u
  end

  prob = SteadyStateDiffEq.SteadyStateProblem(OrdinaryDiffEq.ODEFunction{false}(dudt), z,
                                              ps.model)
  sol = SciMLBase.solve(prob, deq.solver; sensealg=deq.sensealg, deq.kwargs...)
  z_star, st_ = deq.model((sol.u, x), ps.model, st.model)

  if J
    rng = Lux.replicate(st.rng)
    jac_loss = estimate_jacobian_trace(Val(:finite_diff), deq.model, ps.model, st.model,
                                       z_star, x, rng)
  else
    rng = st.rng
    jac_loss = T(0)
  end
  residual = CRC.ignore_derivatives(z_star .- deq.model((z_star, x), ps.model, st.model)[1])

  st = merge(st,
             (model=st_, rng=rng,
              solution=DeepEquilibriumSolution(z_star, z, residual, jac_loss,
                                               sol.destats.nf + 1 + J, sol)))

  return z_star, st
end
