module FastDEQ

using Reexport

@reexport using CUDA
using DataLoaders
using DiffEqBase
using DiffEqCallbacks
@reexport using DiffEqSensitivity
@reexport using Flux
@reexport using FluxExperimental
@reexport using FluxMPI
using Format
using Functors
using Glob
using LearnBase
using LinearAlgebra
using LinearSolve
using MLDataUtils
using MPI
@reexport using OrdinaryDiffEq
using Random
using Requires
using SciMLBase
using Serialization
using SparseArrays
using Statistics
@reexport using SteadyStateDiffEq
using UnPack
@reexport using Zygote

function __init__()
    @require ChemistryFeaturization = "6c925690-434a-421d-aea7-51398c5b007a" begin
        include("models/cgcnn.jl")
        export CrystalGraphCNN
    end
end

abstract type AbstractDeepEquilibriumNetwork end

function Base.show(io::IO, l::AbstractDeepEquilibriumNetwork)
    return print(io, string(typeof(l).name.name), "() ", string(length(l.p)), " Trainable Parameters")
end

Flux.trainable(d::AbstractDeepEquilibriumNetwork) = (d.p,)

Base.deepcopy(op::DiffEqSensitivity.ZygotePullbackMultiplyOperator) = op

abstract type IterativeDEQSolver end

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
include("models/width_stacked_deq.jl")
include("models/mpgnn.jl")

include("dataset.jl")

include("losses.jl")
include("logger.jl")

# DEQ Layers
export DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork, MultiScaleDeepEquilibriumNetwork,
       MultiScaleSkipDeepEquilibriumNetwork, WidthStackedDEQ

# Compositional Layers
export MaterialsProjectCrystalGraphConvNet, MaterialsProjectGraphConv
export DEQChain
export BasicResidualBlock, BranchNet, MultiParallelNet, BasicBottleneckBlock

# For the sanity of experiment code
export get_and_clear_nfe!, get_default_ssadjoint, get_default_dynamicss_solver, get_default_ssrootfind_solver, normal_init

# Text Logging
export PrettyTableLogger

# Again experiment code sanity
export SupervisedLossContainer

# Non Linear Solvers (need to find time to move into a dedicated package with a proper API)
export BroydenSolver, BroydenCache, LimitedMemoryBroydenSolver, LimitedMemoryBroydenCache

end
