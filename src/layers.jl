"""
    DeepEquilibriumSolution(z_star, u₀, residual, jacobian_loss, nfe, solution)

Stores the solution of a DeepEquilibriumNetwork and its variants.

## Fields

  - `z_star`: Steady-State or the value reached due to maxiters
  - `u0`: Initial Condition
  - `residual`: Difference of the ``z^*`` and ``f(z^*, x)``
  - `jacobian_loss`: Jacobian Stabilization Loss (see individual networks to see how it
    can be computed)
  - `nfe`: Number of Function Evaluations
  - `original`: Original Internal Solution
"""
struct DeepEquilibriumSolution  # This is intentionally left untyped to allow updating `st`
    z_star
    u0
    residual
    jacobian_loss
    nfe::Int
    original
end

function CRC.rrule(::Type{<:DeepEquilibriumSolution}, z_star,
        u0, residual, jacobian_loss, nfe, original)
    sol = DeepEquilibriumSolution(z_star, u0, residual, jacobian_loss, nfe, original)
    ∇DeepEquilibriumSolution(::CRC.NoTangent) = ntuple(_ -> CRC.NoTangent(), 7)
    function ∇DeepEquilibriumSolution(∂sol)
        return (CRC.NoTangent(), ∂sol.z_star, ∂sol.u0, ∂sol.residual,
            ∂sol.jacobian_loss, ∂sol.nfe, CRC.NoTangent())
    end
    return sol, ∇DeepEquilibriumSolution
end

function DeepEquilibriumSolution()
    return DeepEquilibriumSolution(ntuple(Returns(nothing), 4)..., 0, nothing)
end

function Base.show(io::IO, sol::DeepEquilibriumSolution)
    println(io, "DeepEquilibriumSolution")
    println(io, " * Initial Guess: ", sprint(print, sol.u0; context=(
        :compact => true, :limit => true)))
    println(io, " * Steady State: ", sprint(print, sol.z_star; context=(
        :compact => true, :limit => true)))
    println(io, " * Residual: ", sprint(print, sol.residual; context=(
        :compact => true, :limit => true)))
    println(io, " * Jacobian Loss: ",
        sprint(print, sol.jacobian_loss; context=(:compact => true, :limit => true)))
    print(io, " * NFE: ", sol.nfe)
end

# Core Model
@concrete struct DeepEquilibriumNetwork <: AbstractLuxContainerLayer{(:model, :init)}
    init
    model
    solver
    jacobian_regularization
    kwargs
    kind <: StaticSymbol
end

const DEQ = DeepEquilibriumNetwork

function LuxCore.initialstates(rng::AbstractRNG, deq::DEQ)
    rng = LuxCore.replicate(rng)
    randn(rng, 1)
    return (; model=LuxCore.initialstates(rng, deq.model), fixed_depth=Val(0),
        init=LuxCore.initialstates(rng, deq.init), solution=DeepEquilibriumSolution(), rng)
end

(deq::DEQ)(x, ps, st::NamedTuple) = deq(x, ps, st, check_unrolled_mode(st))

## Pretraining
function (deq::DEQ)(x, ps, st::NamedTuple, ::Val{true})
    z, st = get_initial_condition(deq, x, ps, st)
    repeated_model = RepeatedLayer(deq.model; repeats=st.fixed_depth)

    z_star, st_ = repeated_model((z, x), ps.model, st.model)
    model = StatefulLuxLayer{true}(deq.model, ps.model, st_)
    resid = CRC.ignore_derivatives(z_star .- model((z_star, x)))

    rng = LuxCore.replicate(st.rng)
    jac_loss = estimate_jacobian_trace(
        LuxOps.getproperty(deq, Val(:jacobian_regularization)), model, z_star, x, rng)

    solution = DeepEquilibriumSolution(
        z_star, z, resid, zero(eltype(x)), _unwrap_val(st.fixed_depth), jac_loss)
    res = split_and_reshape(z_star, LuxOps.getproperty(deq.model, Val(:split_idxs)),
        LuxOps.getproperty(deq.model, Val(:scales)))

    return res, (; st..., model=model.st, solution, rng)
end

function (deq::DEQ)(x, ps, st::NamedTuple, ::Val{false})
    z, st = get_initial_condition(deq, x, ps, st)

    model = StatefulLuxLayer{true}(deq.model, ps.model, st.model)

    dudt = @closure (u, p, t) -> begin
        # The type-assert is needed because of an upstream Lux issue with type stability of
        # conv with Dual numbers
        y = model((u, p.x), p.ps)::typeof(u)
        return y .- u
    end

    prob = construct_prob(deq.kind, ODEFunction{false}(dudt), z, (; ps=ps.model, x))
    alg = normalize_alg(deq)
    termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs))
    sol = solve(prob, alg; sensealg=default_sensealg(prob), abstol=1e-3,
        reltol=1e-3, termination_condition, maxiters=32, deq.kwargs...)
    z_star = get_steady_state(sol)

    rng = LuxCore.replicate(st.rng)
    jac_loss = estimate_jacobian_trace(
        LuxOps.getproperty(deq, Val(:jacobian_regularization)), model, z_star, x, rng)

    solution = DeepEquilibriumSolution(
        z_star, z, LuxOps.getproperty(sol, Val(:resid)), jac_loss, get_nfe(sol), sol)
    res = split_and_reshape(z_star, LuxOps.getproperty(deq.model, Val(:split_idxs)),
        LuxOps.getproperty(deq.model, Val(:scales)))

    return res, (; st..., model=model.st, solution, rng)
end

## Constructors
"""
    DeepEquilibriumNetwork(model, solver; init = missing, jacobian_regularization=nothing,
        problem_type::Type=SteadyStateProblem{false}, kwargs...)

Deep Equilibrium Network as proposed in [baideep2019](@cite) and [pal2022mixing](@cite).

## Arguments

  - `model`: Neural Network.
  - `solver`: Solver for the rootfinding problem. ODE Solvers and Nonlinear Solvers are both
    supported.

## Keyword Arguments

  - `init`: Initial Condition for the rootfinding problem. If `nothing`, the initial
    condition is set to `zero(x)`. If `missing`, the initial condition is set to
    `WrappedFunction(zero)`. In other cases the initial condition is set to
    `init(x, ps, st)`.
  - `jacobian_regularization`: Must be one of `nothing`, `AutoForwardDiff`, `AutoFiniteDiff`
    or `AutoZygote`.
  - `problem_type`: Provides a way to simulate a Vanilla Neural ODE by setting the
    `problem_type` to `ODEProblem`. By default, the problem type is set to
    `SteadyStateProblem`.
  - `kwargs`: Additional Parameters that are directly passed to `SciMLBase.solve`.

## Example

```jldoctest
julia> model = DeepEquilibriumNetwork(
           Parallel(+, Dense(2, 2; use_bias=false), Dense(2, 2; use_bias=false)),
           VCABM3(); verbose=false);

julia> rng = Xoshiro(0);

julia> ps, st = Lux.setup(rng, model);

julia> size(first(model(ones(Float32, 2, 1), ps, st)))
(2, 1)
```

See also: [`SkipDeepEquilibriumNetwork`](@ref), [`MultiScaleDeepEquilibriumNetwork`](@ref),
[`MultiScaleSkipDeepEquilibriumNetwork`](@ref).
"""
function DeepEquilibriumNetwork(
        model, solver; init=missing, jacobian_regularization=nothing,
        problem_type::Type=SteadyStateProblem{false}, kwargs...)
    if init === missing # Regular DEQ
        init = WrappedFunction(Base.Fix1(zeros_init, LuxOps.getproperty(model, Val(:scales))))
    elseif init === nothing # SkipRegDEQ
        init = NoOpLayer()
    elseif !(init isa AbstractLuxLayer)
        error("init::$(typeof(init)) is not a valid input for DeepEquilibriumNetwork.")
    end
    return DeepEquilibriumNetwork(init, model, solver, jacobian_regularization,
        kwargs, problem_type_to_symbol(problem_type))
end

"""
    SkipDeepEquilibriumNetwork(model, [init=nothing,] solver; kwargs...)

Skip Deep Equilibrium Network as proposed in [pal2022mixing](@cite). Alias which creates
a [`DeepEquilibriumNetwork`](@ref) with `init` kwarg set to passed value.
"""
function SkipDeepEquilibriumNetwork(model, init, solver; kwargs...)
    return DeepEquilibriumNetwork(model, solver; init, kwargs...)
end

function SkipDeepEquilibriumNetwork(model, solver; kwargs...)
    return DeepEquilibriumNetwork(model, solver; init=nothing, kwargs...)
end

## MultiScale DEQ
"""
    MultiScaleDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,
        post_fuse_layer::Union{Nothing, Tuple}, solver,
        scales::NTuple{N, NTuple{L, Int64}}; kwargs...)

Multi Scale Deep Equilibrium Network as proposed in [baimultiscale2020](@cite).

## Arguments

  - `main_layers`: Tuple of Neural Networks. Each Neural Network is applied to the
    corresponding scale.
  - `mapping_layers`: Matrix of Neural Networks. Each Neural Network is applied to the
    corresponding scale and the corresponding layer.
  - `post_fuse_layer`: Neural Network applied to the fused output of the main layers.
  - `solver`: Solver for the rootfinding problem. ODE Solvers and Nonlinear Solvers are both
    supported.
  - `scales`: Scales of the Multi Scale DEQ. Each scale is a tuple of integers. The length
    of the tuple is the number of layers in the corresponding main layer.

For keyword arguments, see [`DeepEquilibriumNetwork`](@ref).

## Example

```jldoctest
julia> main_layers = (
           Parallel(+, Dense(4 => 4, tanh; use_bias=false), Dense(4 => 4, tanh; use_bias=false)),
           Dense(3 => 3, tanh), Dense(2 => 2, tanh), Dense(1 => 1, tanh));

julia> mapping_layers = [NoOpLayer() Dense(4 => 3, tanh) Dense(4 => 2, tanh) Dense(4 => 1, tanh);
                         Dense(3 => 4, tanh) NoOpLayer() Dense(3 => 2, tanh) Dense(3 => 1, tanh);
                         Dense(2 => 4, tanh) Dense(2 => 3, tanh) NoOpLayer() Dense(2 => 1, tanh);
                         Dense(1 => 4, tanh) Dense(1 => 3, tanh) Dense(1 => 2, tanh) NoOpLayer()];

julia> model = MultiScaleDeepEquilibriumNetwork(
           main_layers, mapping_layers, nothing, NewtonRaphson(), ((4,), (3,), (2,), (1,)));

julia> rng = Xoshiro(0);

julia> ps, st = Lux.setup(rng, model);

julia> x = rand(rng, Float32, 4, 12);

julia> size.(first(model(x, ps, st)))
((4, 12), (3, 12), (2, 12), (1, 12))
```
"""
function MultiScaleDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,
        post_fuse_layer::Union{Nothing, Tuple}, solver, scales; kwargs...)
    l1 = Parallel(nothing, main_layers...)
    l2 = BranchLayer(Parallel.(+, map(x -> tuple(x...), eachrow(mapping_layers))...)...)

    scales = Val(scales)
    split_idxs = Val(Tuple(vcat(0, cumsum(prod.(_unwrap_val(scales)))...)))

    if post_fuse_layer === nothing
        model = MultiScaleInputLayer(Chain(l1, l2), split_idxs, scales)
    else
        model = MultiScaleInputLayer(Chain(l1, l2, Parallel(nothing, post_fuse_layer...)), split_idxs, scales)
    end

    return DeepEquilibriumNetwork(model, solver; kwargs...)
end

"""
    MultiScaleSkipDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,
        post_fuse_layer::Union{Nothing, Tuple}, [init = nothing,] solver,
        scales::NTuple{N, NTuple{L, Int64}}; kwargs...)

Skip Multi Scale Deep Equilibrium Network as proposed in [pal2022mixing](@cite). Alias which
creates a [`MultiScaleDeepEquilibriumNetwork`](@ref) with `init` kwarg set to passed value.

If `init` is not passed, it creates a MultiScale Regularized Deep Equilibrium Network.
"""
function MultiScaleSkipDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,
        post_fuse_layer::Union{Nothing, Tuple}, init::Tuple, solver, scales; kwargs...)
    init = Chain(Parallel(nothing, init...), flatten_vcat)
    return MultiScaleDeepEquilibriumNetwork(
        main_layers, mapping_layers, post_fuse_layer, solver, scales; init, kwargs...)
end

function MultiScaleSkipDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,
        post_fuse_layer::Union{Nothing, Tuple}, args...; kwargs...)
    return MultiScaleDeepEquilibriumNetwork(
        main_layers, mapping_layers, post_fuse_layer, args...; init=nothing, kwargs...)
end

"""
    MultiScaleNeuralODE(args...; kwargs...)

Same arguments as [`MultiScaleDeepEquilibriumNetwork`](@ref) but sets `problem_type` to
`ODEProblem{false}`.
"""
function MultiScaleNeuralODE(args...; kwargs...)
    return MultiScaleDeepEquilibriumNetwork(args...; kwargs..., problem_type=ODEProblem{false})
end

## Generate Initial Condition
function get_initial_condition(deq::DEQ{NoOpLayer}, x, ps, st)
    zₓ = zeros_init(LuxOps.getproperty(deq.model, Val(:scales)), x)
    z, st_ = deq.model((zₓ, x), ps.model, st.model)
    return z, (; st..., model=st_)
end

function get_initial_condition(deq::DEQ, x, ps, st)
    z, st_ = deq.init(x, ps.init, st.init)
    return z, (; st..., init=st_)
end

# Other Layers
@concrete struct MultiScaleInputLayer <: AbstractLuxWrapperLayer{:model}
    n <: StaticInt
    model <: AbstractLuxLayer
    split_idxs
    scales
end

function MultiScaleInputLayer(model, split_idxs, scales::Val{S}) where {S}
    return MultiScaleInputLayer(static(length(S)), model, split_idxs, scales)
end

@generated function (m::MultiScaleInputLayer{N})(z, ps, st) where {N}
    inputs = (:((u_[1], x)), (:(u_[$i]) for i in 2:known(N))...)
    return quote
        u, x = z
        u_ = split_and_reshape(u, m.split_idxs, m.scales)
        u_res, st = LuxCore.apply(m.model, ($(inputs...),), ps, st)
        return flatten_vcat(u_res), st
    end
end
