import ChainRulesCore as CRC
import Lux
import MLUtils
import OrdinaryDiffEq
import SciMLBase
import Static
import SteadyStateDiffEq

@generated function _evaluate_unrolled_mdeq(model, z_star::NTuple{N}, x, ps, st,
                                            ::Val{depth}) where {N, depth}
  calls = []
  for _ in 1:depth
    push!(calls, :((z_star, st) = model(((z_star[1], x), z_star[2:($N)]...), ps, st)))
  end
  push!(calls, :(return z_star, st))
  return Expr(:block, calls...)
end

"""
    MultiScaleDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,
                                     post_fuse_layer::Union{Nothing,Tuple}, solver, scales;
                                     sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10),
                                     kwargs...)

Multiscale Deep Equilibrium Network as proposed in [baimultiscale2020](@cite)

## Arguments

  - `main_layers`: Tuple of Neural Networks. The first network needs to take a tuple of 2
    arrays, the other ones only take 1 input.
  - `mapping_layers`: Matrix of Neural Networks. The ``(i, j)^{th}`` network takes the
    output of ``i^{th}`` `main_layer` and passes it to the ``j^{th}`` `main_layer`.
  - `post_fuse_layer`: Tuple of Neural Networks. Each of the scales is passed through this
    layer.
  - `solver`: Solver for the optimization problem (See: [`ContinuousDEQSolver`](@ref) &
    [`DiscreteDEQSolver`](@ref)).
  - `scales`: Output scales.
  - `sensealg`: See [`DeepEquilibriumAdjoint`](@ref).
  - `kwargs`: Additional Parameters that are directly passed to `SciMLBase.solve`.

## Example

```@example
import DeepEquilibriumNetworks as DEQs
import Lux
import OrdinaryDiffEq
import Random

main_layers = (Lux.Parallel(+, Lux.Dense(4, 4, tanh), Lux.Dense(4, 4, tanh)),
               Lux.Dense(3, 3, tanh), Lux.Dense(2, 2, tanh), Lux.Dense(1, 1, tanh))

mapping_layers = [Lux.NoOpLayer() Lux.Dense(4, 3, tanh) Lux.Dense(4, 2, tanh) Lux.Dense(4, 1, tanh);
                  Lux.Dense(3, 4, tanh) Lux.NoOpLayer() Lux.Dense(3, 2, tanh) Lux.Dense(3, 1, tanh);
                  Lux.Dense(2, 4, tanh) Lux.Dense(2, 3, tanh) Lux.NoOpLayer() Lux.Dense(2, 1, tanh);
                  Lux.Dense(1, 4, tanh) Lux.Dense(1, 3, tanh) Lux.Dense(1, 2, tanh) Lux.NoOpLayer()]

solver = DEQs.ContinuousDEQSolver(OrdinaryDiffEq.VCABM3(); abstol=0.01f0, reltol=0.01f0)

model = DEQs.MultiScaleDeepEquilibriumNetwork(main_layers, mapping_layers, nothing, solver,
                                              ((4,), (3,), (2,), (1,)))

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)
x = rand(rng, Float32, 4, 1)

model(x, ps, st)
```

See also: [`DeepEquilibriumNetwork`](@ref), [`SkipDeepEquilibriumNetwork`](@ref), [`MultiScaleSkipDeepEquilibriumNetwork`](@ref)
"""
struct MultiScaleDeepEquilibriumNetwork{N, Sc, M, A, S, K} <: AbstractDeepEquilibriumNetwork
  model::M
  solver::A
  sensealg::S
  scales::Sc
  kwargs::K
end

function Lux.initialstates(rng::Random.AbstractRNG, deq::MultiScaleDeepEquilibriumNetwork)
  return (model=Lux.initialstates(rng, deq.model),
          split_idxs=Static.static(Tuple(vcat(0, cumsum(prod.(deq.scales))...))),
          fixed_depth=Val(0), initial_condition=zeros(Float32, 1, 1), solution=nothing)
end

function MultiScaleDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,
                                          post_fuse_layer::Union{Nothing, Tuple}, solver,
                                          scales::NTuple{N, NTuple{L, Int64}};
                                          sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10),
                                          kwargs...) where {N, L}
  l1 = Lux.Parallel(nothing, main_layers...)
  l2 = Lux.BranchLayer(Lux.Parallel.(+,
                                     map(x -> tuple(x...), eachrow(mapping_layers))...)...)
  if post_fuse_layer === nothing
    model = Lux.Chain(l1, l2)
  else
    model = Lux.Chain(l1, l2, Lux.Parallel(nothing, post_fuse_layer...))
  end
  scales = Static.static(scales)
  return MultiScaleDeepEquilibriumNetwork{N, typeof(scales), typeof(model), typeof(solver),
                                          typeof(sensealg), typeof(kwargs)}(model, solver,
                                                                            sensealg,
                                                                            scales, kwargs)
end

@generated function _get_initial_condition_mdeq(::S, x::AbstractArray{T, N},
                                                st::NamedTuple{fields}) where {S, T, N,
                                                                               fields}
  scales = Static.known(S)
  sz = sum(prod.(scales))
  calls = []
  if :initial_condition ∈ fields
    push!(calls, :(u0 = st[:initial_condition]))
    push!(calls, :(($sz, size(x, $N)) == size(u0) && return u0, st))
  end
  push!(calls, :(u0 = fill!(similar(x, $(sz), size(x, N)), $(T(0)))))
  push!(calls, :(st = merge(st, (initial_condition=u0,))))
  push!(calls, :(return u0, st))
  return Expr(:block, calls...)
end

CRC.@non_differentiable _get_initial_condition_mdeq(::Any...)

function (deq::MultiScaleDeepEquilibriumNetwork{N})(x::AbstractArray{T}, ps,
                                                    st::NamedTuple) where {N, T}
  z, st = _get_initial_condition_mdeq(deq.scales, x, st)

  if _check_unrolled_mode(st)
    z_star = split_and_reshape(z, st.split_idxs, deq.scales)
    z_star, st_ = _evaluate_unrolled_mdeq(deq.model, z_star, x, ps, st.model,
                                          st.fixed_depth)

    residual = CRC.ignore_derivatives(vcat(MLUtils.flatten.(z_star)...) .-
                                      vcat(MLUtils.flatten.(_evaluate_unrolled_mdeq(deq.model,
                                                                                    z_star,
                                                                                    x, ps,
                                                                                    st_,
                                                                                    Val(1))[1])...))
    solution = DeepEquilibriumSolution(vcat(MLUtils.flatten.(z_star)...), z, residual,
                                       0.0f0, _get_unrolled_depth(st))
    st__ = merge(st, (model=st_, solution=solution))

    return z_star, st__
  end

  st_ = st.model

  function dudt_(u, p, t)
    u_split = split_and_reshape(u, st.split_idxs, deq.scales)
    u_, st_ = deq.model(((u_split[1], x), u_split[2:N]...), p, st_)
    return u_, st_
  end

  dudt(u, p, t) = vcat(MLUtils.flatten.(dudt_(u, p, t)[1])...) .- u

  prob = SteadyStateDiffEq.SteadyStateProblem(OrdinaryDiffEq.ODEFunction{false}(dudt), z,
                                              ps)
  sol = SciMLBase.solve(prob, deq.solver; sensealg=deq.sensealg, deq.kwargs...)
  z_star, st_ = dudt_(sol.u, ps, nothing)

  residual = CRC.ignore_derivatives(dudt(sol.u, ps, nothing))

  solution = DeepEquilibriumSolution(vcat(MLUtils.flatten.(z_star)...), z, residual, 0.0f0,
                                     sol.destats.nf + 1)
  st__ = merge(st, (model=st_, solution=solution))

  return z_star, st__
end

"""
    MultiScaleSkipDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,
                                         post_fuse_layer::Union{Nothing,Tuple},
                                         shortcut_layers::Union{Nothing,Tuple}, solver,
                                         scales;
                                         sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10),
                                         kwargs...)

Multiscale Deep Equilibrium Network as proposed in [baimultiscale2020](@cite) combined with
Skip Deep Equilibrium Network as proposed in [pal2022mixing](@cite).

## Arguments

  - `main_layers`: Tuple of Neural Networks. The first network needs to take a tuple of 2
    arrays, the other ones only take 1 input.
  - `mapping_layers`: Matrix of Neural Networks. The ``(i, j)^{th}`` network takes the
    output of ``i^{th}`` `main_layer` and passes it to the ``j^{th}`` `main_layer`.
  - `post_fuse_layer`: Tuple of Neural Networks. Each of the scales is passed through
    this layer.
  - `shortcut_layers`: Shortcut for the network (pass `nothing` for SkipDEQV2).
  - `solver`: Solver for the optimization problem (See: [`ContinuousDEQSolver`](@ref) &
    [`DiscreteDEQSolver`](@ref)).
  - `scales`: Output scales.
  - `sensealg`: See [`DeepEquilibriumAdjoint`](@ref).
  - `kwargs`: Additional Parameters that are directly passed to `SciMLBase.solve`.

## Example

```@example
import DeepEquilibriumNetworks as DEQs
import Lux
import OrdinaryDiffEq
import Random

# MSkipDEQ
main_layers = (Lux.Parallel(+, Lux.Dense(4, 4, tanh), Lux.Dense(4, 4, tanh)),
               Lux.Dense(3, 3, tanh), Lux.Dense(2, 2, tanh), Lux.Dense(1, 1, tanh))

mapping_layers = [Lux.NoOpLayer() Lux.Dense(4, 3, tanh) Lux.Dense(4, 2, tanh) Lux.Dense(4, 1, tanh);
                  Lux.Dense(3, 4, tanh) Lux.NoOpLayer() Lux.Dense(3, 2, tanh) Lux.Dense(3, 1, tanh);
                  Lux.Dense(2, 4, tanh) Lux.Dense(2, 3, tanh) Lux.NoOpLayer() Lux.Dense(2, 1, tanh);
                  Lux.Dense(1, 4, tanh) Lux.Dense(1, 3, tanh) Lux.Dense(1, 2, tanh) Lux.NoOpLayer()]

solver = DEQs.ContinuousDEQSolver(OrdinaryDiffEq.VCABM3(); abstol=0.01f0, reltol=0.01f0)

shortcut_layers = (Lux.Dense(4, 4, tanh), Lux.Dense(4, 3, tanh), Lux.Dense(4, 2, tanh),
                   Lux.Dense(4, 1, tanh))

model = DEQs.MultiScaleSkipDeepEquilibriumNetwork(main_layers, mapping_layers, nothing,
                                                  shortcut_layers, solver,
                                                  ((4,), (3,), (2,), (1,)))

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)
x = rand(rng, Float32, 4, 2)

model(x, ps, st)

# MSkipDEQV2
model = DEQs.MultiScaleSkipDeepEquilibriumNetwork(main_layers, mapping_layers, nothing,
                                                  nothing, solver, ((4,), (3,), (2,), (1,)))

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)
x = rand(rng, Float32, 4, 2)

model(x, ps, st)
```

See also: [`DeepEquilibriumNetwork`](@ref), [`SkipDeepEquilibriumNetwork`](@ref),
[`MultiScaleDeepEquilibriumNetwork`](@ref)
"""
struct MultiScaleSkipDeepEquilibriumNetwork{N, Sc, M, Sh, A, S, K} <:
       AbstractSkipDeepEquilibriumNetwork
  model::M
  shortcut::Sh
  solver::A
  sensealg::S
  scales::Sc
  kwargs::K
end

function Lux.initialstates(rng::Random.AbstractRNG,
                           deq::MultiScaleSkipDeepEquilibriumNetwork)
  return (model=Lux.initialstates(rng, deq.model),
          shortcut=Lux.initialstates(rng, deq.shortcut),
          split_idxs=Static.static(Tuple(vcat(0, cumsum(prod.(deq.scales))...))),
          fixed_depth=Val(0), initial_condition=zeros(Float32, 1, 1), solution=nothing)
end

function MultiScaleSkipDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,
                                              post_fuse_layer::Union{Nothing, Tuple},
                                              shortcut_layers::Union{Nothing, Tuple},
                                              solver, scales;
                                              sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0,
                                                                              10),
                                              kwargs...)
  l1 = Lux.Parallel(nothing, main_layers...)
  l2 = Lux.BranchLayer(Lux.Parallel.(+,
                                     map(x -> tuple(x...), eachrow(mapping_layers))...)...)
  if post_fuse_layer === nothing
    model = Lux.Chain(l1, l2)
  else
    model = Lux.Chain(l1, l2, Lux.Parallel(nothing, post_fuse_layer...))
  end
  shortcut = shortcut_layers === nothing ? nothing :
             Lux.Parallel(nothing, shortcut_layers...)
  scales = Static.static(scales)
  return MultiScaleSkipDeepEquilibriumNetwork{length(scales), typeof(scales), typeof(model),
                                              typeof(shortcut), typeof(solver),
                                              typeof(sensealg), typeof(kwargs)}(model,
                                                                                shortcut,
                                                                                solver,
                                                                                sensealg,
                                                                                scales,
                                                                                kwargs)
end

function (deq::MultiScaleSkipDeepEquilibriumNetwork{N, Sc, M, Sh})(x::AbstractArray{T}, ps,
                                                                   st::NamedTuple) where {N,
                                                                                          Sc,
                                                                                          M,
                                                                                          Sh,
                                                                                          T}
  if Sh == Nothing
    u0, st_ = _get_initial_condition_mdeq(deq.scales, x, st)
    u0_ = split_and_reshape(u0, st.split_idxs, deq.scales)
    z0, st__ = deq.model(((u0_[1], x), u0_[2:N]...), ps.model, st_.model)
    z = vcat(MLUtils.flatten.(z0)...)
    st = merge(st_, (model=st__,))
  else
    z0, st_ = deq.shortcut(x, ps.shortcut, st.shortcut)
    z = vcat(MLUtils.flatten.(z0)...)
    st = merge(st, (shortcut=st_,))
  end

  if _check_unrolled_mode(st)
    z_star = split_and_reshape(z, st.split_idxs, deq.scales)
    z_star, st_ = _evaluate_unrolled_mdeq(deq.model, z_star, x, ps.model, st.model,
                                          st.fixed_depth)

    residual = CRC.ignore_derivatives(vcat(MLUtils.flatten.(z_star)...) .-
                                      vcat(MLUtils.flatten.(_evaluate_unrolled_mdeq(deq.model,
                                                                                    z_star,
                                                                                    x,
                                                                                    ps.model,
                                                                                    st_,
                                                                                    Val(1))[1])...))
    st__ = merge(st,
                 (model=st_,
                  solution=DeepEquilibriumSolution(vcat(MLUtils.flatten.(z_star)...), z,
                                                   residual, 0.0f0,
                                                   _get_unrolled_depth(st))))

    return z_star, st__
  end

  st_ = st.model

  function dudt_(u, p, t)
    u_split = split_and_reshape(u, st.split_idxs, deq.scales)
    u_, st_ = deq.model(((u_split[1], x), u_split[2:N]...), p, st_)
    return u_, st_
  end

  dudt(u, p, t) = vcat(MLUtils.flatten.(dudt_(u, p, t)[1])...) .- u

  prob = SteadyStateDiffEq.SteadyStateProblem(OrdinaryDiffEq.ODEFunction{false}(dudt), z,
                                              ps.model)
  sol = SciMLBase.solve(prob, deq.solver; sensealg=deq.sensealg, deq.kwargs...)
  z_star, st_ = dudt_(sol.u, ps.model, nothing)

  residual = CRC.ignore_derivatives(dudt(sol.u, ps.model, nothing))

  st__ = merge(st,
               (model=st_,
                solution=DeepEquilibriumSolution(vcat(MLUtils.flatten.(z_star)...), z,
                                                 residual, 0.0f0, sol.destats.nf + 1)))

  return z_star, st__
end
