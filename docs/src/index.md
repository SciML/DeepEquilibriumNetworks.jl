# DeepEquilibriumNetworks: (Fast) Deep Equlibrium Networks

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

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode=PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@raw html
You can also download the 
<a href="
```

```@eval
using TOML
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link = "https://github.com/SciML/" *
       name *
       ".jl/tree/gh-pages/v" *
       version *
       "/assets/Manifest.toml"
```

```@raw html
">manifest</a> file and the
<a href="
```

```@eval
using TOML
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link = "https://github.com/SciML/" *
       name *
       ".jl/tree/gh-pages/v" *
       version *
       "/assets/Project.toml"
```

```@raw html
">project</a> file.
```
