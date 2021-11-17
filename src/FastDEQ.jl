module FastDEQ

using ChainRulesCore
using ChemistryFeaturization
using CUDA
using DataDeps
using DataLoaders
using DiffEqBase
using DiffEqCallbacks
using DiffEqSensitivity
using Flux
using Format
using Functors
using LinearAlgebra
using LinearSolve
using MPI
using MultiScaleArrays
using OrdinaryDiffEq
# using RecursiveArrayTools: ArrayPartition
using SciMLBase
using SparseArrays
using Statistics
using SteadyStateDiffEq
using UnPack
using Zygote

using Flux: hasaffine, ones32, zeros32, _isactive


const is_inside_deq = Ref(false)

is_in_deq() = is_inside_deq[]

@inline function update_is_in_deq(val::Bool)
    is_inside_deq[] = val
end


abstract type AbstractDeepEquilibriumNetwork end

function Base.show(io::IO, l::AbstractDeepEquilibriumNetwork)
    p, _ = Flux.destructure(l)
    print(
        io,
        string(typeof(l).name.name),
        "() ",
        string(length(p)),
        " Trainable Parameters",
    )
end


include("utils.jl")
include("dataloaders.jl")
include("solvers/broyden.jl")
include("solvers/limited_memory_broyden.jl")
include("solvers/linsolve.jl")
include("layers/agn.jl")
include("layers/deq.jl")
include("layers/sdeq.jl")
include("layers/mdeq.jl")
include("layers/smdeq.jl")
include("layers/dropout.jl")
include("layers/normalise.jl")
include("models/chain.jl")
include("models/basics.jl")
include("models/cgcnn.jl")
include("losses.jl")
include("zygote.jl")
include("logger.jl")


export DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork
export MultiScaleDeepEquilibriumNetworkS4,
    MultiScaleSkipDeepEquilibriumNetworkS4
export AGNConv, AGNMaxPool, AGNMeanPool
export VariationalHiddenDropout
export GroupNormV2

export DEQChain, Sequential
export CrystalGraphCNN
export BasicResidualBlock

export batch_graph_data, BatchedAtomicGraph
export reset_mask!
export get_and_clear_nfe!

export PrettyTableLogger

export SupervisedLossContainer

export BroydenCache, broyden
export LimitedMemoryBroydenSolver, LimitedMemoryBroydenCache
export LinSolveKrylovJL

export SingleResolutionFeatures, MultiResolutionFeatures

end
