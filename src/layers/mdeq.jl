@concrete struct MultiScaleInputLayer{N, M <: Lux.AbstractExplicitLayer} <:
                 Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    split_idxs
    scales
end

@truncate_stacktrace MultiScaleInputLayer 1 2

function MultiScaleInputLayer(model, split_idxs, scales::Val{S}) where {S}
    return MultiScaleInputLayer{length(S)}(model, split_idxs, scales)
end

@generated function (m::MultiScaleInputLayer{N})(z, ps, st) where {N}
    inputs = (:((u_[1], x)), (:(u_[$i]) for i in 2:N)...)
    return quote
        u, x = z
        u_ = split_and_reshape(u, m.split_idxs, m.scales)
        u_res, st = m.model(($(inputs...),), ps, st)
        return mapreduce(flatten, vcat, u_res), st
    end
end

"""
    MultiScaleDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,
                                     post_fuse_layer::Union{Nothing,Tuple}, solver, scales;
                                     sensealg=SteadyStateAdjoint(), kwargs...)

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
  - `sensealg`: See `SciMLSensitivity.SteadyStateAdjoint`.
  - `kwargs`: Additional Parameters that are directly passed to `SciMLBase.solve`.

## Example

```@example
using DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq

main_layers = (Parallel(+, Dense(4, 4, tanh), Dense(4, 4, tanh)),
    Dense(3, 3, tanh),
    Dense(2, 2, tanh),
    Dense(1, 1, tanh))

mapping_layers = [NoOpLayer() Dense(4, 3, tanh) Dense(4, 2, tanh) Dense(4, 1, tanh);
    Dense(3, 4, tanh) NoOpLayer() Dense(3, 2, tanh) Dense(3, 1, tanh);
    Dense(2, 4, tanh) Dense(2, 3, tanh) NoOpLayer() Dense(2, 1, tanh);
    Dense(1, 4, tanh) Dense(1, 3, tanh) Dense(1, 2, tanh) NoOpLayer()]

solver = ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0)

model = MultiScaleDeepEquilibriumNetwork(main_layers, mapping_layers, nothing,
    solver, ((4,), (3,), (2,), (1,)); save_everystep=true)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)
x = rand(rng, Float32, 4, 1)

model(x, ps, st)
```

See also: [`DeepEquilibriumNetwork`](@ref), [`SkipDeepEquilibriumNetwork`](@ref), [`MultiScaleSkipDeepEquilibriumNetwork`](@ref)
"""
@concrete struct MultiScaleDeepEquilibriumNetwork{N} <: AbstractDeepEquilibriumNetwork
    model
    solver
    sensealg
    scales
    split_idxs
    kwargs
end

function MultiScaleDeepEquilibriumNetwork(model::MultiScaleInputLayer{N}, args...) where {N}
    return MultiScaleDeepEquilibriumNetwork{N}(model, args...)
end

@truncate_stacktrace MultiScaleDeepEquilibriumNetwork 1 3

function Lux.initialstates(rng::AbstractRNG, deq::MultiScaleDeepEquilibriumNetwork)
    rng = Lux.replicate(rng)
    randn(rng, 1)
    return (; model=Lux.initialstates(rng, deq.model), fixed_depth=Val(0),
        initial_condition=zeros(Float32, 1, 1), solution=nothing, rng)
end

function MultiScaleDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,
    post_fuse_layer::Union{Nothing, Tuple}, solver, scales::NTuple{N, NTuple{L, Int64}};
    sensealg=SteadyStateAdjoint(), kwargs...) where {N, L}
    l1 = Parallel(nothing, main_layers...)
    l2 = BranchLayer(Parallel.(+, map(x -> tuple(x...), eachrow(mapping_layers))...)...)

    scales = Val(scales)
    split_idxs = Val(Tuple(vcat(0, cumsum(prod.(SciMLBase._unwrap_val(scales)))...)))
    if post_fuse_layer === nothing
        model = MultiScaleInputLayer(Chain(l1, l2), split_idxs, scales)
    else
        model = MultiScaleInputLayer(Chain(l1, l2, Parallel(nothing, post_fuse_layer...)),
            split_idxs, scales)
    end

    return MultiScaleDeepEquilibriumNetwork(model, solver, sensealg, scales, split_idxs,
        kwargs)
end

_jacobian_regularization(::MultiScaleDeepEquilibriumNetwork) = false

function _get_initial_condition(deq::MultiScaleDeepEquilibriumNetwork, x, ps, st)
    return _get_zeros_initial_condition_mdeq(deq.scales, x, st)
end

"""
    MultiScaleSkipDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,
                                         post_fuse_layer::Union{Nothing,Tuple},
                                         shortcut_layers::Union{Nothing,Tuple}, solver,
                                         scales; sensealg=SteadyStateAdjoint(), kwargs...)

Multiscale Deep Equilibrium Network as proposed in [baimultiscale2020](@cite) combined with
Skip Deep Equilibrium Network as proposed in [pal2022mixing](@cite).

## Arguments

  - `main_layers`: Tuple of Neural Networks. The first network needs to take a tuple of 2
    arrays, the other ones only take 1 input.
  - `mapping_layers`: Matrix of Neural Networks. The ``(i, j)^{th}`` network takes the
    output of ``i^{th}`` `main_layer` and passes it to the ``j^{th}`` `main_layer`.
  - `post_fuse_layer`: Tuple of Neural Networks. Each of the scales is passed through
    this layer.
  - `shortcut_layers`: Shortcut for the network (pass `nothing` for SkipRegDEQ).
  - `solver`: Solver for the optimization problem (See: [`ContinuousDEQSolver`](@ref) &
    [`DiscreteDEQSolver`](@ref)).
  - `scales`: Output scales.
  - `sensealg`: See `SciMLSensitivity.SteadyStateAdjoint`.
  - `kwargs`: Additional Parameters that are directly passed to `SciMLBase.solve`.

## Example

```@example
using DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq

# MSkipDEQ
main_layers = (Parallel(+, Dense(4, 4, tanh), Dense(4, 4, tanh)),
    Dense(3, 3, tanh),
    Dense(2, 2, tanh),
    Dense(1, 1, tanh))

mapping_layers = [NoOpLayer() Dense(4, 3, tanh) Dense(4, 2, tanh) Dense(4, 1, tanh);
    Dense(3, 4, tanh) NoOpLayer() Dense(3, 2, tanh) Dense(3, 1, tanh);
    Dense(2, 4, tanh) Dense(2, 3, tanh) NoOpLayer() Dense(2, 1, tanh);
    Dense(1, 4, tanh) Dense(1, 3, tanh) Dense(1, 2, tanh) NoOpLayer()]

solver = ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0)

shortcut_layers = (Dense(4, 4, tanh),
    Dense(4, 3, tanh),
    Dense(4, 2, tanh),
    Dense(4, 1, tanh))

model = MultiScaleSkipDeepEquilibriumNetwork(main_layers,
    mapping_layers,
    nothing,
    shortcut_layers,
    solver,
    ((4,), (3,), (2,), (1,));
    save_everystep=true)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)
x = rand(rng, Float32, 4, 2)

model(x, ps, st)

# MSkipRegDEQ
model = MultiScaleSkipDeepEquilibriumNetwork(main_layers,
    mapping_layers,
    nothing,
    nothing,
    solver,
    ((4,), (3,), (2,), (1,));
    save_everystep=true)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)
x = rand(rng, Float32, 4, 2)

model(x, ps, st)
```

See also: [`DeepEquilibriumNetwork`](@ref), [`SkipDeepEquilibriumNetwork`](@ref),
[`MultiScaleDeepEquilibriumNetwork`](@ref)
"""
@concrete struct MultiScaleSkipDeepEquilibriumNetwork{N} <:
                 AbstractSkipDeepEquilibriumNetwork
    model
    shortcut
    solver
    sensealg
    scales
    split_idxs
    kwargs
end

function MultiScaleSkipDeepEquilibriumNetwork(model::MultiScaleInputLayer{N},
    args...) where {N}
    return MultiScaleSkipDeepEquilibriumNetwork{N}(model, args...)
end

@truncate_stacktrace MultiScaleSkipDeepEquilibriumNetwork 1 3 4

function Lux.initialstates(rng::AbstractRNG, deq::MultiScaleSkipDeepEquilibriumNetwork)
    rng = Lux.replicate(rng)
    randn(rng, 1)
    return (; model=Lux.initialstates(rng, deq.model), fixed_depth=Val(0), rng,
        shortcut=Lux.initialstates(rng, deq.shortcut), solution=nothing,
        initial_condition=zeros(Float32, 1, 1))
end

function MultiScaleSkipDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,
    post_fuse_layer::Union{Nothing, Tuple}, shortcut_layers::Union{Nothing, Tuple},
    solver, scales::NTuple{N, NTuple{L, Int64}};
    sensealg=SteadyStateAdjoint(), kwargs...) where {N, L}
    l1 = Parallel(nothing, main_layers...)
    l2 = BranchLayer(Parallel.(+, map(x -> tuple(x...), eachrow(mapping_layers))...)...)
    shortcut = shortcut_layers === nothing ? nothing : Parallel(nothing, shortcut_layers...)
    scales = Val(scales)
    split_idxs = Val(Tuple(vcat(0, cumsum(prod.(SciMLBase._unwrap_val(scales)))...)))
    if post_fuse_layer === nothing
        model = MultiScaleInputLayer(Chain(l1, l2), split_idxs, scales)
    else
        model = MultiScaleInputLayer(Chain(l1, l2, Parallel(nothing, post_fuse_layer...)),
            split_idxs, scales)
    end

    return MultiScaleSkipDeepEquilibriumNetwork(model, shortcut, solver, sensealg,
        scales, split_idxs, kwargs)
end

_jacobian_regularization(::MultiScaleSkipDeepEquilibriumNetwork) = false

function _get_initial_condition(deq::MultiScaleSkipDeepEquilibriumNetwork{N, M, Nothing},
    x, ps, st) where {N, M}
    u0, st = _get_zeros_initial_condition_mdeq(deq.scales, x, st)
    z, st_ = deq.model((u0, x), ps.model, st.model)
    @set! st.model = st_
    return z, st
end

function _get_initial_condition(deq::MultiScaleSkipDeepEquilibriumNetwork, x, ps, st)
    z0, st_ = deq.shortcut(x, ps.shortcut, st.shortcut)
    z = mapreduce(flatten, vcat, z0)
    @set! st.shortcut = st_
    return z, st
end

@concrete struct MultiScaleNeuralODE{N} <: AbstractDeepEquilibriumNetwork
    model
    solver
    sensealg
    scales
    split_idxs
    kwargs
end

@truncate_stacktrace MultiScaleNeuralODE 1 3

function Lux.initialstates(rng::Random.AbstractRNG, node::MultiScaleNeuralODE)
    rng = Lux.replicate(rng)
    randn(rng, 1)
    return (; model=Lux.initialstates(rng, node.model), fixed_depth=Val(0),
        initial_condition=zeros(Float32, 1, 1), solution=nothing, rng)
end

"""
    MultiScaleNeuralODE(main_layers::Tuple, mapping_layers::Matrix,
                        post_fuse_layer::Union{Nothing,Tuple}, solver, scales;
                        sensealg=GaussAdjoint(; autojacvec=ZygoteVJP()), kwargs...)

Multiscale Neural ODE with Input Injection.

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
  - `sensealg`: See `SciMLSensitivity.InterpolatingAdjoint`.
  - `kwargs`: Additional Parameters that are directly passed to `SciMLBase.solve`.

## Example

```@example
using DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq

main_layers = (Parallel(+, Dense(4, 4, tanh), Dense(4, 4, tanh)),
    Dense(3, 3, tanh),
    Dense(2, 2, tanh),
    Dense(1, 1, tanh))

mapping_layers = [NoOpLayer() Dense(4, 3, tanh) Dense(4, 2, tanh) Dense(4, 1, tanh);
    Dense(3, 4, tanh) NoOpLayer() Dense(3, 2, tanh) Dense(3, 1, tanh);
    Dense(2, 4, tanh) Dense(2, 3, tanh) NoOpLayer() Dense(2, 1, tanh);
    Dense(1, 4, tanh) Dense(1, 3, tanh) Dense(1, 2, tanh) NoOpLayer()]

model = MultiScaleNeuralODE(main_layers,
    mapping_layers,
    nothing,
    VCAB3(),
    ((4,), (3,), (2,), (1,));
    save_everystep=true)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)
x = rand(rng, Float32, 4, 1)

model(x, ps, st)
```

See also: [`DeepEquilibriumNetwork`](@ref), [`SkipDeepEquilibriumNetwork`](@ref), [`MultiScaleDeepEquilibriumNetwork`](@ref), [`MultiScaleSkipDeepEquilibriumNetwork`](@ref)
"""
function MultiScaleNeuralODE(main_layers::Tuple, mapping_layers::Matrix,
    post_fuse_layer::Union{Nothing, Tuple}, solver, scales::NTuple{N, NTuple{L, Int64}};
    sensealg=GaussAdjoint(; autojacvec=ZygoteVJP()), kwargs...) where {N, L}
    l1 = Parallel(nothing, main_layers...)
    l2 = BranchLayer(Parallel.(+, map(x -> tuple(x...), eachrow(mapping_layers))...)...)

    scales = Val(scales)
    split_idxs = Val(Tuple(vcat(0, cumsum(prod.(SciMLBase._unwrap_val(scales)))...)))
    if post_fuse_layer === nothing
        model = MultiScaleInputLayer(Chain(l1, l2), split_idxs, scales)
    else
        model = MultiScaleInputLayer(Chain(l1, l2, Parallel(nothing, post_fuse_layer...)),
            split_idxs, scales)
    end

    return MultiScaleNeuralODE{N}(model, solver, sensealg, scales, split_idxs, kwargs)
end

_jacobian_regularization(::MultiScaleNeuralODE) = false

function _get_initial_condition(deq::MultiScaleNeuralODE, x, ps, st)
    return _get_zeros_initial_condition_mdeq(deq.scales, x, st)
end

@inline function _construct_problem(::MultiScaleNeuralODE, dudt, z, ps, x)
    return ODEProblem(ODEFunction{false}(dudt), z, (0.0f0, 1.0f0), (; ps=ps.model, x))
end

@inline _fix_solution_output(::MultiScaleNeuralODE, x) = x[end]

# Shared Functions
@generated function _get_zeros_initial_condition_mdeq(::Val{scales}, x::AbstractArray{T, N},
    st::NamedTuple{fields}) where {scales, T, N, fields}
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

CRC.@non_differentiable _get_zeros_initial_condition_mdeq(::Any...)

@inline function _postprocess_output(deq::Union{MultiScaleDeepEquilibriumNetwork,
        MultiScaleSkipDeepEquilibriumNetwork, MultiScaleNeuralODE}, z_star)
    return split_and_reshape(z_star, deq.split_idxs, deq.scales)
end
