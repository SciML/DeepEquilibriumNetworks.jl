# Deep Equilibrium Models

[baideep2019](@cite) introduced Discrete Deep Equilibrium Models which drives a Discrete
Dynamical System to its steady-state. [pal2022mixing](@cite) extends this framework to
Continuous Dynamical Systems which converge to the steady-stable in a more stable fashion.
For a detailed discussion refer to [pal2022mixing](@cite).

To construct a continuous DEQ, any ODE solver compatible with `DifferentialEquations.jl` API
can be passed as the solver. To construct a discrete DEQ, any root finding algorithm
compatible with `NonlinearSolve.jl` API can be passed as the solver.

## Standard Models

```@docs
DeepEquilibriumNetwork
SkipDeepEquilibriumNetwork
```

## MultiScale Models

```@docs
MultiScaleDeepEquilibriumNetwork
MultiScaleSkipDeepEquilibriumNetwork
MultiScaleNeuralODE
```

## Solution

```@docs
DeepEquilibriumSolution
```
