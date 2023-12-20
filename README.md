# DeepEquilibriumNetworks

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/DeepEquilibriumNetworks/stable/)

[![codecov](https://codecov.io/gh/SciML/DeepEquilibriumNetworks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SciML/DeepEquilibriumNetworks.jl)
[![Build Status](https://github.com/SciML/DeepEquilibriumNetworks.jl/workflows/CI/badge.svg)](https://github.com/SciML/DeepEquilibriumNetworks.jl/actions?query=workflow%3ACI)
[![Build status](https://badge.buildkite.com/d7ce1858c4f89456c2d90e80c9b04b710bd81d7641db0a087c.svg?branch=main)](https://buildkite.com/julialang/deepequilibriumnetworks)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

DeepEquilibriumNetworks.jl is a framework built on top of
[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) and
[Lux.jl](https://lux.csail.mit.edu/) enabling the efficient training and inference for
Deep Equilibrium Networks (Infinitely Deep Neural Networks).

## Installation

```julia
using Pkg
Pkg.add("DeepEquilibriumNetworks")
```

## Quickstart

```julia
using DeepEquilibriumNetworks, Lux, Random, NonlinearSolve, Zygote, SciMLSensitivity
# using LuxCUDA, LuxAMDGPU ## Install and Load for GPU Support. See https://lux.csail.mit.edu/dev/manual/gpu_management

seed = 0
rng = Random.default_rng()
Random.seed!(rng, seed)

model = Chain(Dense(2 => 2),
    DeepEquilibriumNetwork(Parallel(+, Dense(2 => 2; use_bias=false),
            Dense(2 => 2; use_bias=false)), NewtonRaphson()))

gdev = gpu_device()
cdev = cpu_device()

ps, st = Lux.setup(rng, model) |> gdev
x = rand(rng, Float32, 2, 3) |> gdev
y = rand(rng, Float32, 2, 3) |> gdev

model(x, ps, st)

gs = only(Zygote.gradient(p -> sum(abs2, first(model(x, p, st)) .- y), ps))
```

## Citation

If you are using this project for research or other academic purposes consider citing our
paper:

```bibtex
@article{pal2022continuous,
  title={Continuous Deep Equilibrium Models: Training Neural ODEs Faster by Integrating Them to Infinity},
  author={Pal, Avik and Edelman, Alan and Rackauckas, Christopher},
  booktitle={2023 IEEE High Performance Extreme Computing Conference (HPEC)}, 
  year={2023}
}
```

For specific algorithms, check the respective documentations and cite the corresponding
papers.
