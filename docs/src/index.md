# FastDEQ: (Fast) Deep Equlibrium Models

FastDEQ.jl is a framework built on top of [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) and [Flux.jl](https://fluxml.ai) enabling the efficient training and inference for Deep Equilibrium Networks (Infinitely Deep Neural Networks).

## Installation

Currently the package is not registered and requires manually installing a few dependencies. We are working towards upstream fixes which will make installation easier

```julia
] add https://github.com/SciML/DiffEqSensitivity.jl.git#ap/fastdeq
] add https://github.com/avik-pal/FluxExperimental.jl.git#main
] add https://github.com/SciML/FastDEQ.jl
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

## FAQs

#### How do I reproduce the experiments in the paper -- *Mixing Implicit and Explicit Deep Learning with Skip DEQs and Infinite Time Neural ODEs (Continuous DEQs)*?

Check out the `ap/paper` branch for the code corresponding to that paper.

#### Are there some tutorials?

We are working on adding some in the near future. In the meantime, please checkout the `experiments` directory in the `ap/paper` branch. You can also check `test/runtests.jl` for some simple examples.