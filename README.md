# FastDEQ

![Dynamics Overview](assets/dynamics_overview.gif)

## Installation

This package relies on some unreleased packages which need to be manually installed

```julia
] add https://github.com/avik-pal/FluxExperimental.jl.git
] add DiffEqSensitivity#ap/fastdeq
] add https://github.com/avik-pal/Zygote.jl.git#ap/fastdeq
] add https://github.com/avik-pal/FastDEQ.jl
```

## Implemented Models

* `DeepEquilibriumNetwork`: The standard DEQ Layer https://arxiv.org/abs/1909.01377.
* `SkipDeepEquilibriumNetwork`: Our version of faster DEQ Model.
* `MultiScaleDeepEquilibriumNetwork`
* `MultiScaleSkipDeepEquilibriumNetwork`
* `DEQChain`: Use `DEQChain` instead of `Chain` if your model contains a `DEQ` or `SkipDEQ` layer.

## Non-Linear Solvers

* `BroydenSolver`
* `LimitedMemoryBroydenSolver`

## Experiments

1. [polynomial_fitting.jl](experiments/polynomial_fitting.jl) -- Comparing the Performance of SkipDEQ and DEQ when fitting on `y = x^2`
   *  Execute like a normal julia script 
2. [mnist_mdeq.jl](experiments/mnist_mdeq.jl) -- Supervised MNIST Classification using MDEQ
   *  See [mnist_mdeq.sh](scripts/mnist_mdeq.sh) for an example slurm script. It uses paths and details corresponding to our internal cluster so it is very likely that you need to modify the parameters before running the script. (If you are running this on MIT Supercloud just modify the `cd ...` line)
3. [cifar10_deq.jl](experiments/cifar10_deq.jl) -- Supervised CIFAR-10 Classification using MDEQ
   *  See [cifar_mdeq.sh](scripts/cifar_mdeq.sh) -- Similar to MNIST this will require modifications. (If you are running this on MIT Supercloud just modify the `cd ...` line)

### Some Notes:

1. The `MPI` binaries provided by binary builder will probably not be GPU aware. If it isn't `FluxMPI` will display a warning. All the code will work but the transfer will most likely be slower. My personal recommendation is to compile `openmpi` with `ucx` and `cuda` support and setup `MPI.jl` to use this compiled binary. It is likely to cost about 30 mins to setup but will save massive pains.
2. The slurm scripts are written assuming the 2 GPUs are V100s with 32GB GPU Memory. In that is not the case, reduce `mpiexecjl -np <value>` to a lower number (1 process will approximately take 10GB memory).


## Troubleshooting

1. `libhdf5.so` not found: If you have admin priviledges, just install hdf5 for your system. Else install `h5py` using Conda and export a new environment `JULIA_HDF5_PATH="<path to (ana|mini)conda>/lib"`. Next do `]build` in the Julia REPL.
