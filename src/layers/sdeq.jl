"""
    SkipDeepEquilibriumNetwork(model, shortcut, solver; p=nothing, jacobian_regularization::Bool=false,
                               sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10), kwargs...)
    SkipDeepEquilibriumNetwork(model, solver; p=nothing, jacobian_regularization::Bool=false,
                               sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10), kwargs...)

Skip Deep Equilibrium Network as proposed in [pal2022mixing](@cite)

## Arguments

* `model`: Explicit Neural Network which takes 2 inputs
* `shortcut`: Shortcut for the network (If not given, then we create SkipDEQV2)
* `solver`: Solver for the optimization problem (See: [`ContinuousDEQSolver`](@ref) & [`DiscreteDEQSolver`](@ref))
* `jacobian_regularization`: If true, Jacobian Loss is computed and stored in the [`DeepEquilibriumSolution`](@ref)
* `p`: Optional parameters for the `model`
* `sensealg`: See [`SteadyStateAdjoint`](@ref)
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

model(rand(Float32, 2, 1))

# SkipDEQV2
model = SkipDeepEquilibriumNetwork(
    Parallel(
        +,
        Dense(2, 2; bias=false),
        Dense(2, 2; bias=false)
    ),
    ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0)
)

model(rand(Float32, 2, 1))
```

See also: [`DeepEquilibriumNetwork`](@ref), [`MultiScaleDeepEquilibriumNetwork`](@ref), [`MultiScaleSkipDeepEquilibriumNetwork`](@ref)
"""
struct SkipDeepEquilibriumNetwork{M,S,J,P,RE1,RE2,A,Se,K} <: AbstractDeepEquilibriumNetwork
    jacobian_regularization::Bool
    model::M
    shortcut::S
    p::P
    re1::RE1
    re2::RE2
    split_idx::Int
    solver::A
    kwargs::K
    sensealg::Se
    stats::DEQTrainingStats

    function SkipDeepEquilibriumNetwork(jacobian_regularization, model, shortcut, p, re1, re2,
                                        split_idx, solver, kwargs, sensealg, stats)
        p1, re1 = destructure_parameters(model)
        split_idx = length(p1)
        p2, re2 = shortcut === nothing ? ((eltype(p1))[], nothing) : destructure_parameters(shortcut)

        p = p === nothing ? vcat(p1, p2) : eltype(p1).(p)

        return new{typeof(model),typeof(shortcut),jacobian_regularization,typeof(p),typeof(re1),
                   typeof(re2),typeof(solver),typeof(sensealg),typeof(kwargs)}(jacobian_regularization, model, shortcut, p,
                                                                               re1, re2, split_idx, solver, kwargs,
                                                                               sensealg, stats)
    end
end

Flux.@functor SkipDeepEquilibriumNetwork

function Base.show(io::IO, l::SkipDeepEquilibriumNetwork{M,S,J}) where {M,S,J}
    shortcut_ps = l.split_idx == length(l.p) ? 0 : length(l.p) - l.split_idx
    return print(io, "SkipDeepEquilibriumNetwork(jacobian_regularization = $J, ",
                 "shortcut_parameter_count = $shortcut_ps) ", string(length(l.p)), " Trainable Parameters")
end

function SkipDeepEquilibriumNetwork(model, shortcut, solver; p=nothing, jacobian_regularization::Bool=false,
                                    sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10), kwargs...)
    return SkipDeepEquilibriumNetwork(jacobian_regularization, model, shortcut, p, nothing,
                                      nothing, 0, solver, kwargs, sensealg, DEQTrainingStats(0))
end

function SkipDeepEquilibriumNetwork(model, solver; p=nothing, jacobian_regularization::Bool=false,
                                    sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10), kwargs...)
    return SkipDeepEquilibriumNetwork(jacobian_regularization, model, nothing, p, nothing,
                                      nothing, 0, solver, kwargs, sensealg, DEQTrainingStats(0))
end

function (deq::SkipDeepEquilibriumNetwork)(x::AbstractArray{T}) where {T}
    p1, p2 = deq.p[1:(deq.split_idx)], deq.p[(deq.split_idx + 1):end]
    z = deq.re2(p2)(x)::typeof(x)

    current_nfe = deq.stats.nfe

    # Dummy call to ensure that mask is generated
    Zygote.@ignore _ = deq.re1(p1)(z, x)

    z_star = solve_steady_state_problem(deq.re1, p1, x, z, deq.sensealg, deq.solver; dudt=nothing,
                                        update_nfe=() -> (deq.stats.nfe += 1), deq.kwargs...)

    jac_loss = (deq.jacobian_regularization ? compute_deq_jacobian_loss(deq.re1, p1, z_star, x) : T(0)) ::T

    residual = Zygote.@ignore z_star .- deq.re1(p1)(z_star, x)

    return z_star, DeepEquilibriumSolution(z_star, z, residual, jac_loss, deq.stats.nfe - current_nfe)
end

function (deq::SkipDeepEquilibriumNetwork{M,Nothing})(x::AbstractArray{T}) where {M,T}
    z = deq.re1(deq.p)(zero(x), x)::typeof(x)

    current_nfe = deq.stats.nfe

    z_star = solve_steady_state_problem(deq.re1, deq.p, x, z, deq.sensealg, deq.solver; dudt=nothing,
                                        update_nfe=() -> (deq.stats.nfe += 1), deq.kwargs...)
    
    jac_loss = (deq.jacobian_regularization ? compute_deq_jacobian_loss(deq.re1, deq.p, z_star, x) : T(0)) ::T

    residual = Zygote.@ignore z_star .- deq.re1(deq.p)(z_star, x)

    return z_star, DeepEquilibriumSolution(z_star, z, residual, jac_loss, deq.stats.nfe - current_nfe)
end
