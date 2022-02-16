"""
    DeepEquilibriumNetwork(model, solver; jacobian_regularization::Bool=false,
                           p=nothing, sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10),
                           kwargs...)

Deep Equilibrium Network as proposed in [baideep2019](@cite)

## Arguments

* `model`: Explicit Neural Network which takes 2 inputs
* `solver`: Solver for the optimization problem (See: [`ContinuousDEQSolver`](@ref) & [`DiscreteDEQSolver`](@ref))
* `jacobian_regularization`: If true, Jacobian Loss is computed and stored in the [`DeepEquilibriumSolution`](@ref)
* `p`: Optional parameters for the `model`
* `sensealg`: See [`SteadyStateAdjoint`](@ref)
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

model(rand(Float32, 2, 1))
```

See also: [`SkipDeepEquilibriumNetwork`](@ref), [`MultiScaleDeepEquilibriumNetwork`](@ref), [`MultiScaleSkipDeepEquilibriumNetwork`](@ref)
"""
struct DeepEquilibriumNetwork{J,M,P,RE,A,S,K} <: AbstractDeepEquilibriumNetwork
    jacobian_regularization::Bool
    model::M
    p::P
    re::RE
    solver::A
    kwargs::K
    sensealg::S
    stats::DEQTrainingStats

    function DeepEquilibriumNetwork(jacobian_regularization, model, p, re, solver, kwargs, sensealg, stats)
        _p, re = destructure_parameters(model)
        p = p === nothing ? _p : convert(typeof(_p), p)

        return new{jacobian_regularization,typeof(model),typeof(p),typeof(re),typeof(solver),
                   typeof(sensealg),typeof(kwargs)}(jacobian_regularization, model, p, re,
                                                    solver, kwargs, sensealg, stats)
    end
end

Flux.@functor DeepEquilibriumNetwork

function Base.show(io::IO, l::DeepEquilibriumNetwork{J}) where {J}
    return print(io, "DeepEquilibriumNetwork(jacobian_regularization = $J) ",
                 string(length(l.p)), " Trainable Parameters")
end

function DeepEquilibriumNetwork(model, solver; jacobian_regularization::Bool=false,
                                p=nothing, sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10), kwargs...)
    return DeepEquilibriumNetwork(jacobian_regularization, model, p, nothing, solver,
                                  kwargs, sensealg, DEQTrainingStats(0))
end

function (deq::DeepEquilibriumNetwork)(x::AbstractArray{T}) where {T}
    z = zero(x)
    Zygote.@ignore deq.re(deq.p)(z, x)

    current_nfe = deq.stats.nfe

    z_star = solve_steady_state_problem(deq.re, deq.p, x, z, deq.sensealg, deq.solver; dudt=nothing,
                                        update_nfe=() -> (deq.stats.nfe += 1), deq.kwargs...)

    jac_loss = (deq.jacobian_regularization ? compute_deq_jacobian_loss(deq.re, deq.p, z_star, x) : T(0))::T

    residual = Zygote.@ignore z_star .- deq.re(deq.p)(z_star, x)

    return z_star, DeepEquilibriumSolution(z_star, z, residual, jac_loss, deq.stats.nfe - current_nfe)
end
