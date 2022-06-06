# DeepEquilibriumNetworks: (Fast) Deep Equlibrium Networks

DeepEquilibriumNetworks.jl is a framework built on top of [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) and [Lux.jl](https://lux.csail.mit.edu/dev/) enabling the efficient training and inference for Deep Equilibrium Networks (Infinitely Deep Neural Networks).

## Installation

```julia
] add DeepEquilibriumNetworks
```

## Quickstart

```julia
using Lux, DeepEquilibriumNetworks, Random

seed = 0
rng = Random.default_rng()
Random.seed!(rng, seed)

model = DEQChain(Dense(2, 2),
                 DeepEquilibriumNetwork(Parallel(+, Dense(2, 2; bias=false),
                                                 Dense(2, 2; bias=false)),
                                        ContinuousDEQSolver(; abstol=0.1f0,
                                                            reltol=0.1f0,
                                                            abstol_termination=0.1f0,
                                                            reltol_termination=0.1f0)))

ps, st = gpu.(Lux.setup(rng, model))
x = gpu(rand(rng, Float32, 2, 1))
y = gpu(rand(rng, Float32, 2, 1))

gs = gradient(p -> sum(abs2, model(x, p, st)[1][1] .- y), ps)[1]
```

## Citation

If you are using this project for research or other academic purposes consider citing our paper:

```bibtex
@misc{pal2022mixing,
      title={Mixing Implicit and Explicit Deep Learning with Skip DEQs and Infinite Time Neural ODEs (Continuous DEQs)}, 
      author={Avik Pal and Alan Edelman and Christopher Rackauckas},
      year={2022},
      eprint={2201.12240},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

For specific algorithms, check the respective documentations and cite the corresponding papers.