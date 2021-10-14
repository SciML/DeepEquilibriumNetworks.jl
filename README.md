# FastDEQ

## Implemented Models

* DeepEquilibriumNetwork
* SkipDeepEquilibriumNetwork

## Non-Linear Solvers (GPU Compatible)

To be ported to NonLinearSolve.jl

* Broyden

## Linear Solvers (GPU Compatible)

* `LinSolveKrylovJL`: Wraps `KrylovJL` from LinearSolve.jl to have a compatible API for DiffEqSensitivity


## Experiments

1. [polynomial_fitting.jl](experiments/polynomial_fitting.jl) -- Comparing the Performance of SkipDEQ and DEQ when fitting on $y = x^2$
2. [mnist_deq.jl](experiments/mnist_deq.jl) -- Supervised MNIST Classification (using ConvNets + DEQ)
