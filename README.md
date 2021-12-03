# FastDEQ

## Implemented Models

* `DeepEquilibriumNetwork`: The standard DEQ Layer https://arxiv.org/abs/1909.01377.
* `SkipDeepEquilibriumNetwork`: Our version of faster DEQ Model.
* `DEQChain`: Use `DEQChain` instead of `Chain` if your model contains a `DEQ` or `SkipDEQ` layer.

## Non-Linear Solvers (GPU Compatible)

* `BroydenSolver`
* `LimitedMemoryBroydenSolver`

## Experiments

1. [polynomial_fitting.jl](experiments/polynomial_fitting.jl) -- Comparing the Performance of SkipDEQ and DEQ when fitting on `y = x^2`
2. [mnist_deq.jl](experiments/mnist_deq.jl) -- Supervised MNIST Classification (using ConvNets + DEQ)
3. [mnist_mdeq.jl](experiments/mnist_mdeq.jl) -- Supervised MNIST Classification using MDEQ (Quite slow to train at the moment)

## Troubleshooting

1. `libhdf5.so` not found: If you have admin priviledges, just install hdf5 for your system. Else install `h5py` using Conda and export a new environment `JULIA_HDF5_PATH="<path to (ana|mini)conda>/lib"`. Next do `]build` in the Julia REPL.

## TODOs

-[x] `Flux.@functor` messes up the parameters of models. The easiest fix is to manually define the parameters and device transfer functions (`cpu` / `gpu`). (We are using a custom `destructure` which only returns the parameters and not every array)
-[ ] `compute_deq_jacobian_loss` doesn't work with models which internally use `destructure`. *[Low Priority]*
-[ ] Update the DEQs to return the jacobian loss
