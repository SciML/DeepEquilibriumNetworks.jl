module FastDEQ

using CUDA, DiffEqBase, DiffEqCallbacks, DiffEqSensitivity, Flux, FluxExperimental, LinearAlgebra, LinearSolve,
      OrdinaryDiffEq, SciMLBase, Statistics, SteadyStateDiffEq, UnPack, Zygote

abstract type AbstractDeepEquilibriumNetwork end

function Base.show(io::IO, l::AbstractDeepEquilibriumNetwork)
    return print(io, string(typeof(l).name.name), "() ", string(length(l.p)), " Trainable Parameters")
end

Flux.trainable(d::AbstractDeepEquilibriumNetwork) = (d.p,)

Base.deepcopy(op::DiffEqSensitivity.ZygotePullbackMultiplyOperator) = op

include("solve.jl")
include("utils.jl")

include("solvers/broyden.jl")
include("solvers/limited_memory_broyden.jl")

include("models/basics.jl")

include("layers/jacobian_stabilization.jl")
include("layers/utils.jl")
include("layers/deq.jl")
include("layers/sdeq.jl")
include("layers/mdeq.jl")
include("layers/smdeq.jl")

include("models/chain.jl")

include("losses.jl")

# DEQ Solvers
export ContinuousDEQSolver, DiscreteDEQSolver, BroydenSolver, LimitedMemoryBroydenSolver

# Utils
export NormalInitializer, SteadyStateAdjoint, get_and_clear_nfe!, compute_deq_jacobian_loss, DeepEquilibriumSolution, SupervisedLossContainer

# Layers
export MultiParallelNet

# DEQ Layers
export DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork, MultiScaleDeepEquilibriumNetwork, MultiScaleSkipDeepEquilibriumNetwork
export DEQChain

end
