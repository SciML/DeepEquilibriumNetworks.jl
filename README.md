# DeepEquilibriumNetworks


[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/DeepEquilibriumNetworks/stable/)

[![codecov](https://codecov.io/gh/SciML/DeepEquilibriumNetworks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SciML/DeepEquilibriumNetworks.jl)
[![Build Status](https://github.com/SciML/DeepEquilibriumNetworks.jl/workflows/CI/badge.svg)](https://github.com/SciML/DeepEquilibriumNetworks.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

DeepEquilibriumNetworks.jl is a framework built on top of
[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) and
[Lux.jl](https://docs.sciml.ai/Lux/stable/) enabling the efficient training and inference for
Deep Equilibrium Networks (Infinitely Deep Neural Networks).

## Installation

```julia
using Pkg
Pkg.add("DeepEquilibriumNetworks")
```

## Quickstart

```julia
import DeepEquilibriumNetworks as DEQs
import Lux
import Random
import Zygote

seed = 0
rng = Random.default_rng()
Random.seed!(rng, seed)

model = Lux.Chain(Lux.Dense(2, 2),
                  DEQs.DeepEquilibriumNetwork(Lux.Parallel(+, Lux.Dense(2, 2; bias=false),
                                                           Lux.Dense(2, 2; bias=false)),
                                              DEQs.ContinuousDEQSolver(; abstol=0.1f0,
                                                                       reltol=0.1f0,
                                                                       abstol_termination=0.1f0,
                                                                       reltol_termination=0.1f0)))

ps, st = gpu.(Lux.setup(rng, model))
x = gpu(rand(rng, Float32, 2, 1))
y = gpu(rand(rng, Float32, 2, 1))

gs = Zygote.gradient(p -> sum(abs2, model(x, p, st)[1][1] .- y), ps)[1]
```

## Citation

If you are using this project for research or other academic purposes consider citing our
paper:

```bibtex
@misc{pal2022mixing,
  title={Mixing Implicit and Explicit Deep Learning with Skip DEQs and Infinite Time Neural
         ODEs (Continuous DEQs)}, 
  author={Avik Pal and Alan Edelman and Christopher Rackauckas},
  year={2022},
  eprint={2201.12240},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

For specific algorithms, check the respective documentations and cite the corresponding
papers.
