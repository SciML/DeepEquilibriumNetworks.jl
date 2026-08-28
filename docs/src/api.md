# Deep Equilibrium Models

[baideep2019](@cite) introduced Discrete Deep Equilibrium Models which drives a Discrete
Dynamical System to its steady-state. [pal2022mixing](@cite) extends this framework to
Continuous Dynamical Systems which converge to the steady-stable in a more stable fashion.
For a detailed discussion refer to [pal2022mixing](@cite).

To construct a continuous DEQ, any ODE solver compatible with `DifferentialEquations.jl` API
can be passed as the solver. To construct a discrete DEQ, any root finding algorithm
compatible with `NonlinearSolve.jl` API can be passed as the solver.

## Choosing a Solver

### Root Finding Algorithms

Using Root Finding Algorithms give fast convergence when possible, but these methods also
tend to be unstable. If you must use a root finding algorithm, we recommend using:

 1. `NewtonRaphson` or `TrustRegion` for small models
 2. `LimitedMemoryBroyden` for large Deep Learning applications (with well-conditioned
    Jacobians)
 3. `NewtonRaphson(; linsolve = KrylovJL_GMRES())` for cases when Broyden methods fail

Note that Krylov Methods rely on efficient VJPs which are not available for all Lux models.
If you think this is causing a performance regression, please open an issue in
[Lux.jl](https://github.com/LuxDL/Lux.jl).

### ODE Solvers

Using ODE Solvers give slower convergence, but are more stable. We generally recommend these
methods over root finding algorithms. If you use implicit ODE solvers, remember to use
Krylov linear solvers, see OrdinaryDiffEq.jl documentation for these. For most cases, we
recommend:

 1. `VCAB3()` for high tolerance problems
 2. `Tsit5()` for high tolerance problems where `VCAB3()` fails
 3. In all other cases, follow the recommendation given in [OrdinaryDiffEq.jl](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#ode_solve) documentation

### Sensitivity Analysis

This package does not override SciMLSensitivity.jl's automatic `sensealg` choice.
`solve` is called with `sensealg = nothing` unless a `sensealg` is passed through the
layer constructor kwargs.

 1. For the out-of-place `SteadyStateProblem`s constructed here, SciMLSensitivity selects
    `SteadyStateAdjoint(autodiff = false, autojacvec = ZygoteVJP())`. Pass
    `sensealg = SteadyStateAdjoint(; linsolve = LUFactorization())` for small models if
    the automatic linear solver is a poor fit.
 2. For `MultiScaleNeuralODE` (functor parameters), SciMLSensitivity selects
    `GaussAdjoint(; autojacvec = ZygoteVJP())`. A faster alternative is
    `BacksolveAdjoint(; autojacvec = ZygoteVJP())`, with the usual stability caveats.
    See the [SciMLSensitivity.jl](https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/#Choosing-a-Sensitivity-Algorithm)
    documentation.

## Public API

```@autodocs
Modules = [DeepEquilibriumNetworks]
Private = false
```
