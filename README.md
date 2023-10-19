# DeepEquilibriumNetworks

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/DeepEquilibriumNetworks/stable/)

[![codecov](https://codecov.io/gh/SciML/DeepEquilibriumNetworks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SciML/DeepEquilibriumNetworks.jl)
[![Build Status](https://github.com/SciML/DeepEquilibriumNetworks.jl/workflows/CI/badge.svg)](https://github.com/SciML/DeepEquilibriumNetworks.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
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
using DeepEquilibriumNetworks, Lux, Random, Zygote
# using LuxCUDA, LuxAMDGPU ## Install and Load for GPU Support

seed = 0
rng = Random.default_rng()
Random.seed!(rng, seed)

model = Chain(Dense(2 => 2),
    DeepEquilibriumNetwork(Parallel(+,
            Dense(2 => 2; use_bias=false),
            Dense(2 => 2; use_bias=false)),
        ContinuousDEQSolver(; abstol=0.1f0, reltol=0.1f0, abstol_termination=0.1f0,
            reltol_termination=0.1f0);
        save_everystep=true))

gdev = gpu_device()
cdev = cpu_device()

ps, st = Lux.setup(rng, model) |> gdev
x = rand(rng, Float32, 2, 1) |> gdev
y = rand(rng, Float32, 2, 1) |> gdev

model(x, ps, st)

gs = only(Zygote.gradient(p -> sum(abs2, first(first(model(x, p, st))) .- y), ps))
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
