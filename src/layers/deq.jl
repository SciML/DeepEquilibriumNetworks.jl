"""
    DeepEquilibriumNetwork(model, solver; jacobian_regularization::Bool=false, sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10), kwargs...)

Deep Equilibrium Network as proposed in [baideep2019](@cite)

## Arguments

* `model`: Neural Network
* `solver`: Solver for the optimization problem (See: [`ContinuousDEQSolver`](@ref) & [`DiscreteDEQSolver`](@ref))
* `jacobian_regularization`: If true, Jacobian Loss is computed and stored in the [`DeepEquilibriumSolution`](@ref)
* `sensealg`: See [`DeepEquilibriumAdjoint`](@ref)
* `kwargs`: Additional Parameters that are directly passed to `solve`

## Example

```julia
model = DeepEquilibriumNetwork(
    Parallel(
        +,
        Dense(2, 2; bias=false),
        Dense(2, 2; bias=false)
    ),
    ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0)
)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

model(rand(Float32, 2, 1), ps, st)
```

See also: [`SkipDeepEquilibriumNetwork`](@ref), [`MultiScaleDeepEquilibriumNetwork`](@ref), [`MultiScaleSkipDeepEquilibriumNetwork`](@ref)
"""
struct DeepEquilibriumNetwork{J,M,A,S,K} <: AbstractDeepEquilibriumNetwork
    model::M
    solver::A
    sensealg::S
    kwargs::K
end

function DeepEquilibriumNetwork(
    model, solver; jacobian_regularization::Bool=false, sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10), kwargs...
)
    return DeepEquilibriumNetwork{jacobian_regularization,typeof(model),typeof(solver),typeof(sensealg),typeof(kwargs)}(
        model, solver, sensealg, kwargs
    )
end

function (deq::DeepEquilibriumNetwork{J})(
    x::AbstractArray{T}, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
) where {J,T}
    z = zero(x)

    if check_unrolled_mode(st)
        # Pretraining without Fixed Point Solving
        st_ = st.model
        z_star = z
        for _ in 1:get_unrolled_depth(st)
            z_star, st_ = deq.model((z_star, x), ps, st_)
        end

        residual = ignore_derivatives(z_star .- deq.model((z_star, x), ps, st.model)[1])
        @set! st.model = Lux.update_state(st_, :update_mask, Val(true))

        return (z_star, DeepEquilibriumSolution(z_star, z, residual, 0.0f0, get_unrolled_depth(st))), st
    end

    st_ = st.model

    function dudt(u, p, t)
        u_, st_ = deq.model((u, x), p, st_)
        return u_ .- u
    end

    prob = SteadyStateProblem(ODEFunction{false}(dudt), z, ps)
    sol = solve(prob, deq.solver; sensealg=deq.sensealg, deq.kwargs...)
    z_star, st_ = deq.model((sol.u, x), ps, st.model)

    jac_loss = (J ? compute_deq_jacobian_loss(deq.model, ps, st.model, z_star, x) : T(0))
    residual = ignore_derivatives(z_star .- deq.model((z_star, x), ps, st.model)[1])

    @set! st.model = Lux.update_state(st_, :update_mask, Val(true))

    return (z_star, DeepEquilibriumSolution(z_star, z, residual, jac_loss, sol.destats.nf + 1 + J)), st
end


"""
    SkipDeepEquilibriumNetwork(model, shortcut, solver; jacobian_regularization::Bool=false, sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10), kwargs...)

Skip Deep Equilibrium Network as proposed in [pal2022mixing](@cite)

## Arguments

* `model`: Neural Network
* `shortcut`: Shortcut for the network (pass `nothing` for SkipDEQV2)
* `solver`: Solver for the optimization problem (See: [`ContinuousDEQSolver`](@ref) & [`DiscreteDEQSolver`](@ref))
* `jacobian_regularization`: If true, Jacobian Loss is computed and stored in the [`DeepEquilibriumSolution`](@ref)
* `sensealg`: See [`DeepEquilibriumAdjoint`](@ref)
* `kwargs`: Additional Parameters that are directly passed to `solve`

## Example

```julia
# SkipDEQ
model = SkipDeepEquilibriumNetwork(
    Parallel(
        +,
        Dense(2, 2; bias=false),
        Dense(2, 2; bias=false)
    ),
    Dense(2, 2),
    ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0)
)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

model(rand(Float32, 2, 1), ps, st)

# SkipDEQV2
model = SkipDeepEquilibriumNetwork(
    Parallel(
        +,
        Dense(2, 2; bias=false),
        Dense(2, 2; bias=false)
    ),
    nothing,
    ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0)
)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

model(rand(Float32, 2, 1), ps, st)
```

See also: [`DeepEquilibriumNetwork`](@ref), [`MultiScaleDeepEquilibriumNetwork`](@ref), [`MultiScaleSkipDeepEquilibriumNetwork`](@ref)
"""
struct SkipDeepEquilibriumNetwork{J,M,Sh,A,S,K} <: AbstractSkipDeepEquilibriumNetwork
    model::M
    shortcut::Sh
    solver::A
    sensealg::S
    kwargs::K
end

function SkipDeepEquilibriumNetwork(
    model,
    shortcut,
    solver;
    jacobian_regularization::Bool=false,
    sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10),
    kwargs...,
)
    return SkipDeepEquilibriumNetwork{
        jacobian_regularization,typeof(model),typeof(shortcut),typeof(solver),typeof(sensealg),typeof(kwargs)
    }(
        model, shortcut, solver, sensealg, kwargs
    )
end

function (deq::SkipDeepEquilibriumNetwork{J,M,S})(
    x::AbstractArray{T}, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
) where {J,M,S,T}
    z, st__ = if S == Nothing
        deq.model((zero(x), x), ps.model, st.model)
    else
        deq.shortcut(x, ps.shortcut, st.shortcut)
    end
    @set! st.shortcut = st__

    if check_unrolled_mode(st)
        # Pretraining without Fixed Point Solving
        st_ = st.model
        z_star = z
        for _ in 1:get_unrolled_depth(st)
            z_star, st_ = deq.model((z_star, x), ps.model, st_)
        end

        residual = ignore_derivatives(z_star .- deq.model((z_star, x), ps.model, st.model)[1])
        @set! st.model = Lux.update_state(st_, :update_mask, Val(true))

        return (z_star, DeepEquilibriumSolution(z_star, z, residual, 0.0f0, get_unrolled_depth(st))), st
    end

    st_ = st.model

    function dudt(u, p, t)
        u_, st_ = deq.model((u, x), p, st_)
        return u_ .- u
    end

    prob = SteadyStateProblem(ODEFunction{false}(dudt), z, ps.model)
    sol = solve(prob, deq.solver; sensealg=deq.sensealg, deq.kwargs...)
    z_star, st_ = deq.model((sol.u, x), ps.model, st.model)

    jac_loss = (J ? compute_deq_jacobian_loss(deq.model, ps.model, st.model, z_star, x) : T(0))
    residual = ignore_derivatives(z_star .- deq.model((z_star, x), ps.model, st.model)[1])

    @set! st.model = Lux.update_state(st_, :update_mask, Val(true))

    return (z_star, DeepEquilibriumSolution(z_star, z, residual, jac_loss, sol.destats.nf + 1 + J)), st
end
