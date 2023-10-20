var documenterSearchIndex = {"docs":
[{"location":"references/#References","page":"References","title":"References","text":"","category":"section"},{"location":"references/","page":"References","title":"References","text":"<div class=\"citation canonical\"><ul><li>\n<div id=\"baideep2019\">Bai, S.; Kolter, J. Z. and Koltun, V. (2019). <a href='http://arxiv.org/abs/1909.01377'><i>Deep Equilibrium Models</i></a>, arXiv:1909.01377 [cs, stat], arXiv: 1909.01377.</div>\n</li><li>\n<div id=\"baimultiscale2020\">Bai, S.; Koltun, V. and Kolter, J. Z. (2020). <a href='http://arxiv.org/abs/2006.08656'><i>Multiscale Deep Equilibrium Models</i></a>, arXiv:2006.08656 [cs, stat], arXiv: 2006.08656.</div>\n</li><li>\n<div id=\"pal2022mixing\">Pal, A.; Edelman, A. and Rackauckas, C. (2022). <i>Mixing implicit and explicit deep learning with skip DEQs and infinite time neural odes (continuous DEQs)</i>. Training <b>4</b>, 5.</div>\n</li>\n</ul></div>","category":"page"},{"location":"manual/deqs/#Deep-Equilibrium-Models","page":"DEQ Layers","title":"Deep Equilibrium Models","text":"","category":"section"},{"location":"manual/deqs/#Standard-Models","page":"DEQ Layers","title":"Standard Models","text":"","category":"section"},{"location":"manual/deqs/","page":"DEQ Layers","title":"DEQ Layers","text":"DeepEquilibriumNetwork\nSkipDeepEquilibriumNetwork","category":"page"},{"location":"manual/deqs/#DeepEquilibriumNetworks.DeepEquilibriumNetwork","page":"DEQ Layers","title":"DeepEquilibriumNetworks.DeepEquilibriumNetwork","text":"DeepEquilibriumNetwork(model, solver; jacobian_regularization::Bool=false,\n                       sensealg=SteadyStateAdjoint(), kwargs...)\n\nDeep Equilibrium Network as proposed in (Bai *et al.*, 2019).\n\nArguments\n\nmodel:Lux Neural Network.\nsolver: Solver for the optimization problem (See: ContinuousDEQSolver & DiscreteDEQSolver).\njacobian_regularization: If true, Jacobian Loss is computed and stored in the DeepEquilibriumSolution.\nsensealg: See SciMLSensitivity.SteadyStateAdjoint.\nkwargs: Additional Parameters that are directly passed to SciMLBase.solve.\n\nExample\n\nusing DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq\n\nmodel = DeepEquilibriumNetwork(Parallel(+,\n        Dense(2, 2; use_bias=false),\n        Dense(2, 2; use_bias=false)),\n    ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0); save_everystep=true)\n\nrng = Random.default_rng()\nps, st = Lux.setup(rng, model)\n\nmodel(rand(rng, Float32, 2, 1), ps, st)\n\nSee also: SkipDeepEquilibriumNetwork, MultiScaleDeepEquilibriumNetwork, MultiScaleSkipDeepEquilibriumNetwork.\n\n\n\n\n\n","category":"type"},{"location":"manual/deqs/#DeepEquilibriumNetworks.SkipDeepEquilibriumNetwork","page":"DEQ Layers","title":"DeepEquilibriumNetworks.SkipDeepEquilibriumNetwork","text":"SkipDeepEquilibriumNetwork(model, shortcut, solver; jacobian_regularization::Bool=false,\n                           sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10), kwargs...)\n\nSkip Deep Equilibrium Network as proposed in (Pal *et al.*, 2022)\n\nArguments\n\nmodel: Neural Network.\nshortcut: Shortcut for the network (pass nothing for SkipRegDEQ).\nsolver: Solver for the optimization problem (See: ContinuousDEQSolver & DiscreteDEQSolver).\njacobian_regularization: If true, Jacobian Loss is computed and stored in the DeepEquilibriumSolution.\nsensealg: See SciMLSensitivity.SteadyStateAdjoint.\nkwargs: Additional Parameters that are directly passed to SciMLBase.solve.\n\nExample\n\nusing DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq\n\n## SkipDEQ\nmodel = SkipDeepEquilibriumNetwork(Parallel(+,\n        Dense(2, 2; use_bias=false),\n        Dense(2, 2; use_bias=false)),\n    Dense(2, 2),\n    ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0); save_everystep=true)\n\nrng = Random.default_rng()\nps, st = Lux.setup(rng, model)\n\nmodel(rand(rng, Float32, 2, 1), ps, st)\n\n## SkipRegDEQ\nmodel = SkipDeepEquilibriumNetwork(Parallel(+,\n        Dense(2, 2; use_bias=false),\n        Dense(2, 2; use_bias=false)),\n    nothing,\n    ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0); save_everystep=true)\n\nrng = Random.default_rng()\nps, st = Lux.setup(rng, model)\n\nmodel(rand(rng, Float32, 2, 1), ps, st)\n\nSee also: DeepEquilibriumNetwork, MultiScaleDeepEquilibriumNetwork, MultiScaleSkipDeepEquilibriumNetwork\n\n\n\n\n\n","category":"type"},{"location":"manual/deqs/#MultiScale-Models","page":"DEQ Layers","title":"MultiScale Models","text":"","category":"section"},{"location":"manual/deqs/","page":"DEQ Layers","title":"DEQ Layers","text":"MultiScaleDeepEquilibriumNetwork\nMultiScaleSkipDeepEquilibriumNetwork\nMultiScaleNeuralODE","category":"page"},{"location":"manual/deqs/#DeepEquilibriumNetworks.MultiScaleDeepEquilibriumNetwork","page":"DEQ Layers","title":"DeepEquilibriumNetworks.MultiScaleDeepEquilibriumNetwork","text":"MultiScaleDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,\n                                 post_fuse_layer::Union{Nothing,Tuple}, solver, scales;\n                                 sensealg=SteadyStateAdjoint(), kwargs...)\n\nMultiscale Deep Equilibrium Network as proposed in (Bai *et al.*, 2020)\n\nArguments\n\nmain_layers: Tuple of Neural Networks. The first network needs to take a tuple of 2 arrays, the other ones only take 1 input.\nmapping_layers: Matrix of Neural Networks. The (i j)^th network takes the output of i^th main_layer and passes it to the j^th main_layer.\npost_fuse_layer: Tuple of Neural Networks. Each of the scales is passed through this layer.\nsolver: Solver for the optimization problem (See: ContinuousDEQSolver & DiscreteDEQSolver).\nscales: Output scales.\nsensealg: See SciMLSensitivity.SteadyStateAdjoint.\nkwargs: Additional Parameters that are directly passed to SciMLBase.solve.\n\nExample\n\nusing DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq\n\nmain_layers = (Parallel(+, Dense(4, 4, tanh), Dense(4, 4, tanh)),\n    Dense(3, 3, tanh),\n    Dense(2, 2, tanh),\n    Dense(1, 1, tanh))\n\nmapping_layers = [NoOpLayer() Dense(4, 3, tanh) Dense(4, 2, tanh) Dense(4, 1, tanh);\n    Dense(3, 4, tanh) NoOpLayer() Dense(3, 2, tanh) Dense(3, 1, tanh);\n    Dense(2, 4, tanh) Dense(2, 3, tanh) NoOpLayer() Dense(2, 1, tanh);\n    Dense(1, 4, tanh) Dense(1, 3, tanh) Dense(1, 2, tanh) NoOpLayer()]\n\nsolver = ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0)\n\nmodel = MultiScaleDeepEquilibriumNetwork(main_layers, mapping_layers, nothing,\n    solver, ((4,), (3,), (2,), (1,)); save_everystep=true)\n\nrng = Random.default_rng()\nps, st = Lux.setup(rng, model)\nx = rand(rng, Float32, 4, 1)\n\nmodel(x, ps, st)\n\nSee also: DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork, MultiScaleSkipDeepEquilibriumNetwork\n\n\n\n\n\n","category":"type"},{"location":"manual/deqs/#DeepEquilibriumNetworks.MultiScaleSkipDeepEquilibriumNetwork","page":"DEQ Layers","title":"DeepEquilibriumNetworks.MultiScaleSkipDeepEquilibriumNetwork","text":"MultiScaleSkipDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix,\n                                     post_fuse_layer::Union{Nothing,Tuple},\n                                     shortcut_layers::Union{Nothing,Tuple}, solver,\n                                     scales; sensealg=SteadyStateAdjoint(), kwargs...)\n\nMultiscale Deep Equilibrium Network as proposed in (Bai *et al.*, 2020) combined with Skip Deep Equilibrium Network as proposed in (Pal *et al.*, 2022).\n\nArguments\n\nmain_layers: Tuple of Neural Networks. The first network needs to take a tuple of 2 arrays, the other ones only take 1 input.\nmapping_layers: Matrix of Neural Networks. The (i j)^th network takes the output of i^th main_layer and passes it to the j^th main_layer.\npost_fuse_layer: Tuple of Neural Networks. Each of the scales is passed through this layer.\nshortcut_layers: Shortcut for the network (pass nothing for SkipRegDEQ).\nsolver: Solver for the optimization problem (See: ContinuousDEQSolver & DiscreteDEQSolver).\nscales: Output scales.\nsensealg: See SciMLSensitivity.SteadyStateAdjoint.\nkwargs: Additional Parameters that are directly passed to SciMLBase.solve.\n\nExample\n\nusing DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq\n\n# MSkipDEQ\nmain_layers = (Parallel(+, Dense(4, 4, tanh), Dense(4, 4, tanh)),\n    Dense(3, 3, tanh),\n    Dense(2, 2, tanh),\n    Dense(1, 1, tanh))\n\nmapping_layers = [NoOpLayer() Dense(4, 3, tanh) Dense(4, 2, tanh) Dense(4, 1, tanh);\n    Dense(3, 4, tanh) NoOpLayer() Dense(3, 2, tanh) Dense(3, 1, tanh);\n    Dense(2, 4, tanh) Dense(2, 3, tanh) NoOpLayer() Dense(2, 1, tanh);\n    Dense(1, 4, tanh) Dense(1, 3, tanh) Dense(1, 2, tanh) NoOpLayer()]\n\nsolver = ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0)\n\nshortcut_layers = (Dense(4, 4, tanh),\n    Dense(4, 3, tanh),\n    Dense(4, 2, tanh),\n    Dense(4, 1, tanh))\n\nmodel = MultiScaleSkipDeepEquilibriumNetwork(main_layers,\n    mapping_layers,\n    nothing,\n    shortcut_layers,\n    solver,\n    ((4,), (3,), (2,), (1,));\n    save_everystep=true)\n\nrng = Random.default_rng()\nps, st = Lux.setup(rng, model)\nx = rand(rng, Float32, 4, 2)\n\nmodel(x, ps, st)\n\n# MSkipRegDEQ\nmodel = MultiScaleSkipDeepEquilibriumNetwork(main_layers,\n    mapping_layers,\n    nothing,\n    nothing,\n    solver,\n    ((4,), (3,), (2,), (1,));\n    save_everystep=true)\n\nrng = Random.default_rng()\nps, st = Lux.setup(rng, model)\nx = rand(rng, Float32, 4, 2)\n\nmodel(x, ps, st)\n\nSee also: DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork, MultiScaleDeepEquilibriumNetwork\n\n\n\n\n\n","category":"type"},{"location":"manual/deqs/#DeepEquilibriumNetworks.MultiScaleNeuralODE","page":"DEQ Layers","title":"DeepEquilibriumNetworks.MultiScaleNeuralODE","text":"MultiScaleNeuralODE(main_layers::Tuple, mapping_layers::Matrix,\n                    post_fuse_layer::Union{Nothing,Tuple}, solver, scales;\n                    sensealg=GaussAdjoint(; autojacvec=ZygoteVJP()), kwargs...)\n\nMultiscale Neural ODE with Input Injection.\n\nArguments\n\nmain_layers: Tuple of Neural Networks. The first network needs to take a tuple of 2 arrays, the other ones only take 1 input.\nmapping_layers: Matrix of Neural Networks. The (i j)^th network takes the output of i^th main_layer and passes it to the j^th main_layer.\npost_fuse_layer: Tuple of Neural Networks. Each of the scales is passed through this layer.\nsolver: Solver for the optimization problem (See: ContinuousDEQSolver & DiscreteDEQSolver).\nscales: Output scales.\nsensealg: See SciMLSensitivity.InterpolatingAdjoint.\nkwargs: Additional Parameters that are directly passed to SciMLBase.solve.\n\nExample\n\nusing DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq\n\nmain_layers = (Parallel(+, Dense(4, 4, tanh), Dense(4, 4, tanh)),\n    Dense(3, 3, tanh),\n    Dense(2, 2, tanh),\n    Dense(1, 1, tanh))\n\nmapping_layers = [NoOpLayer() Dense(4, 3, tanh) Dense(4, 2, tanh) Dense(4, 1, tanh);\n    Dense(3, 4, tanh) NoOpLayer() Dense(3, 2, tanh) Dense(3, 1, tanh);\n    Dense(2, 4, tanh) Dense(2, 3, tanh) NoOpLayer() Dense(2, 1, tanh);\n    Dense(1, 4, tanh) Dense(1, 3, tanh) Dense(1, 2, tanh) NoOpLayer()]\n\nmodel = MultiScaleNeuralODE(main_layers,\n    mapping_layers,\n    nothing,\n    VCAB3(),\n    ((4,), (3,), (2,), (1,));\n    save_everystep=true)\n\nrng = Random.default_rng()\nps, st = Lux.setup(rng, model)\nx = rand(rng, Float32, 4, 1)\n\nmodel(x, ps, st)\n\nSee also: DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork, MultiScaleDeepEquilibriumNetwork, MultiScaleSkipDeepEquilibriumNetwork\n\n\n\n\n\n","category":"type"},{"location":"manual/misc/#Miscellaneous","page":"Miscellaneous","title":"Miscellaneous","text":"","category":"section"},{"location":"manual/misc/","page":"Miscellaneous","title":"Miscellaneous","text":"DeepEquilibriumSolution\nEquilibriumSolution\nDeepEquilibriumNetworks.split_and_reshape\nDeepEquilibriumNetworks.estimate_jacobian_trace","category":"page"},{"location":"manual/misc/#DeepEquilibriumNetworks.DeepEquilibriumSolution","page":"Miscellaneous","title":"DeepEquilibriumNetworks.DeepEquilibriumSolution","text":"DeepEquilibriumSolution(z_star, u₀, residual, jacobian_loss, nfe)\n\nStores the solution of a DeepEquilibriumNetwork and its variants.\n\nFields\n\nz_star: Steady-State or the value reached due to maxiters\nu0: Initial Condition\nresidual: Difference of the z^* and f(z^* x)\njacobian_loss: Jacobian Stabilization Loss (see individual networks to see how it can be computed).\nnfe: Number of Function Evaluations\n\n\n\n\n\n","category":"type"},{"location":"manual/misc/#DeepEquilibriumNetworks.EquilibriumSolution","page":"Miscellaneous","title":"DeepEquilibriumNetworks.EquilibriumSolution","text":"EquilibriumSolution\n\nWraps the solution of a SteadyStateProblem using either ContinuousDEQSolver or DiscreteDEQSolver. This is mostly an internal implementation detail, which allows proper dispatch during adjoint computation without type piracy.\n\n\n\n\n\n","category":"type"},{"location":"manual/misc/#DeepEquilibriumNetworks.split_and_reshape","page":"Miscellaneous","title":"DeepEquilibriumNetworks.split_and_reshape","text":"split_and_reshape(x::AbstractMatrix, ::Val{idxs}, ::Val{shapes}) where {idxs, shapes}\n\nSplits up the AbstractMatrix into chunks and reshapes them.\n\nArguments\n\nx: Matrix to be split up.\nSidxs: Indices to partition the array at. (must be a Val type).\nSshapes: Shapes to reshape the split the arrays. (must be a Val type).\n\nExample\n\nusing DeepEquilibriumNetworks, Static\n\nx1 = ones(Float32, 4, 4)\nx2 = fill!(zeros(Float32, 2, 4), 0.5f0)\nx3 = zeros(Float32, 1, 4)\n\nx = vcat(x1, x2, x3)\nsplit_idxs = Val(cumsum((0, size(x1, 1), size(x2, 1), size(x3, 1))))\nshapes = Val((size(x1, 1), size(x2, 1), size(x3, 1)))\n\nDEQs.split_and_reshape(x, split_idxs, shapes)\n\n\n\n\n\n","category":"function"},{"location":"manual/misc/#DeepEquilibriumNetworks.estimate_jacobian_trace","page":"Miscellaneous","title":"DeepEquilibriumNetworks.estimate_jacobian_trace","text":"estimate_jacobian_trace(::Val{mode}, model::Lux.AbstractExplicitLayer, ps,\n                        st::NamedTuple, z::AbstractArray, x::AbstractArray,\n                        rng::Random.AbstractRNG)\n\nEstimates the trace of the jacobian matrix wrt z.\n\nArguments\n\nmode: Options: reverse and finite_diff\nmodel: A Lux Neural Network mapping 2 equal sized arrays to a same sized array. This convention is not checked, and if violated will lead to errors.\nps: Parameters of model.\nst: States of model.\nz: Input wrt the Jacobian is computed.\nx: Other Input to model.\nrng: PRNG. Note that this object is mutated by this function.\n\nReturns\n\nStochastic Estimate of the trace of the Jacobian.\n\n\n\n\n\n","category":"function"},{"location":"manual/solvers/#Dynamical-System-Variants","page":"Dynamical Systems","title":"Dynamical System Variants","text":"","category":"section"},{"location":"manual/solvers/","page":"Dynamical Systems","title":"Dynamical Systems","text":"(Bai *et al.*, 2019) introduced Discrete Deep Equilibrium Models which drives a Discrete Dynamical System to its steady-state. (Pal *et al.*, 2022) extends this framework to Continuous Dynamical Systems which converge to the steady-stable in a more stable fashion. For a detailed discussion refer to (Pal *et al.*, 2022).","category":"page"},{"location":"manual/solvers/#Continuous-DEQs","page":"Dynamical Systems","title":"Continuous DEQs","text":"","category":"section"},{"location":"manual/solvers/","page":"Dynamical Systems","title":"Dynamical Systems","text":"ContinuousDEQSolver","category":"page"},{"location":"manual/solvers/#DeepEquilibriumNetworks.ContinuousDEQSolver","page":"Dynamical Systems","title":"DeepEquilibriumNetworks.ContinuousDEQSolver","text":"ContinuousDEQSolver(alg=VCAB3(); mode=NLSolveTerminationMode.RelSafeBest,\n    abstol=1.0f-8, reltol=1.0f-6, abstol_termination=abstol, reltol_termination=reltol,\n    tspan=Inf32, kwargs...)\n\nSolver for Continuous DEQ Problem (Pal *et al.*, 2022). Effectively a wrapper around DynamicSS with more sensible defaults for DEQs.\n\nArguments\n\nalg: Algorithm to solve the ODEProblem. (Default: VCAB3())\nmode: Termination Mode of the solver. See the documentation for NLSolveTerminationCondition for more information. (Default: NLSolveTerminationMode.RelSafeBest)\nabstol: Absolute tolerance for time stepping. (Default: 1f-8)\nreltol: Relative tolerance for time stepping. (Default: 1f-6)\nabstol_termination: Absolute tolerance for termination. (Default: abstol)\nreltol_termination: Relative tolerance for termination. (Default: reltol)\ntspan: Time span. Users should not change this value, instead control termination through maxiters in solve. (Default: Inf32)\nkwargs: Additional Parameters that are directly passed to NLSolveTerminationCondition.\n\nSee also: DiscreteDEQSolver\n\n\n\n\n\n","category":"type"},{"location":"manual/solvers/#Discrete-DEQs","page":"Dynamical Systems","title":"Discrete DEQs","text":"","category":"section"},{"location":"manual/solvers/","page":"Dynamical Systems","title":"Dynamical Systems","text":"DiscreteDEQSolver","category":"page"},{"location":"manual/solvers/#DeepEquilibriumNetworks.DiscreteDEQSolver","page":"Dynamical Systems","title":"DeepEquilibriumNetworks.DiscreteDEQSolver","text":"DiscreteDEQSolver(alg = LBroyden(; batched=true,\n    termination_condition=NLSolveTerminationCondition(NLSolveTerminationMode.RelSafe;\n        abstol=1.0f-8, reltol=1.0f-6))\n\nSolver for Discrete DEQ Problem (Bai *et al.*, 2019). Similar to SSrootfind but provides more flexibility needed for solving DEQ problems.\n\nArguments\n\nalg: Algorithm to solve the Nonlinear Problem. (Default: LBroyden)\n\nSee also: ContinuousDEQSolver\n\n\n\n\n\n","category":"type"},{"location":"#DeepEquilibriumNetworks:-(Fast)-Deep-Equilibrium-Networks","page":"Home","title":"DeepEquilibriumNetworks: (Fast) Deep Equilibrium Networks","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"DeepEquilibriumNetworks.jl is a framework built on top of DifferentialEquations.jl and Lux.jl, enabling the efficient training and inference for Deep Equilibrium Networks (Infinitely Deep Neural Networks).","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install DeepEquilibriumNetworks.jl, use the Julia package manager:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(\"DeepEquilibriumNetworks\")","category":"page"},{"location":"#Quick-start","page":"Home","title":"Quick-start","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using DeepEquilibriumNetworks, Lux, Random, Zygote\n# using LuxCUDA, LuxAMDGPU ## Install and Load for GPU Support\n\nseed = 0\nrng = Random.default_rng()\nRandom.seed!(rng, seed)\nmodel = Chain(Dense(2 => 2),\n    DeepEquilibriumNetwork(Parallel(+,\n            Dense(2 => 2; use_bias=false),\n            Dense(2 => 2; use_bias=false)),\n        ContinuousDEQSolver(; abstol=0.1f0, reltol=0.1f0, abstol_termination=0.1f0,\n            reltol_termination=0.1f0);\n        save_everystep=true))\n\ngdev = gpu_device()\ncdev = cpu_device()\n\nps, st = Lux.setup(rng, model) |> gdev\nx = rand(rng, Float32, 2, 1) |> gdev\ny = rand(rng, Float32, 2, 1) |> gdev\n\nmodel(x, ps, st)\n\ngs = only(Zygote.gradient(p -> sum(abs2, first(first(model(x, p, st))) .- y), ps))","category":"page"},{"location":"#Citation","page":"Home","title":"Citation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you are using this project for research or other academic purposes, consider citing our paper:","category":"page"},{"location":"","page":"Home","title":"Home","text":"@misc{pal2022mixing,\n  title={Mixing Implicit and Explicit Deep Learning with Skip DEQs and Infinite Time Neural\n         ODEs (Continuous DEQs)}, \n  author={Avik Pal and Alan Edelman and Christopher Rackauckas},\n  year={2022},\n  eprint={2201.12240},\n  archivePrefix={arXiv},\n  primaryClass={cs.LG}\n}","category":"page"},{"location":"","page":"Home","title":"Home","text":"For specific algorithms, check the respective documentations and cite the corresponding papers.","category":"page"},{"location":"#Contributing","page":"Home","title":"Contributing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Please refer to the SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages for guidance on PRs, issues, and other matters relating to contributing to SciML.\nSee the SciML Style Guide for common coding practices and other style decisions.\nThere are a few community forums:\nThe #diffeq-bridged and #sciml-bridged channels in the Julia Slack\nThe #diffeq-bridged and #sciml-bridged channels in the Julia Zulip\nOn the Julia Discourse forums\nSee also SciML Community page","category":"page"},{"location":"#Reproducibility","page":"Home","title":"Reproducibility","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg # hide\nPkg.status() # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"<details><summary>and using this machine and Julia version.</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using InteractiveUtils # hide\nversioninfo() # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg # hide\nPkg.status(; mode=PKGMODE_MANIFEST) # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"</details>","category":"page"},{"location":"","page":"Home","title":"Home","text":"using TOML\nusing Markdown\nversion = TOML.parse(read(\"../../Project.toml\", String))[\"version\"]\nname = TOML.parse(read(\"../../Project.toml\", String))[\"name\"]\nlink_manifest = \"https://github.com/SciML/\" *\n                name *\n                \".jl/tree/gh-pages/v\" *\n                version *\n                \"/assets/Manifest.toml\"\nlink_project = \"https://github.com/SciML/\" *\n               name *\n               \".jl/tree/gh-pages/v\" *\n               version *\n               \"/assets/Project.toml\"\nMarkdown.parse(\"\"\"You can also download the\n[manifest]($link_manifest)\nfile and the\n[project]($link_project)\nfile.\n\"\"\")","category":"page"}]
}
