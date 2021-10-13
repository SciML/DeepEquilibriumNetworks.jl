# FastDEQ

## Implemented Models

* DeepEquilibriumNetwork
* SkipDeepEquilibriumNetwork

## Non-Linear Solvers (GPU Compatible)

To be ported to NonLinearSolve.jl

* Broyden

## Linear Solvers (GPU Compatible)

* `LinSolveKrylovJL`: Wraps `KrylovJL` from LinearSolve.jl to have a compatible API for DiffEqSensitivity
