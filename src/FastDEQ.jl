module FastDEQ

using ChainRulesCore
using CUDA
using DiffEqBase
using DiffEqCallbacks
using DiffEqSensitivity
using Flux
using Functors
using LinearAlgebra
using LinearSolve
using MultiScaleArrays
using OrdinaryDiffEq
# using RecursiveArrayTools: ArrayPartition
using SciMLBase
using Statistics
using SteadyStateDiffEq
using UnPack
using Zygote


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
include("solvers/broyden.jl")
include("solvers/linsolve.jl")
include("layers/agn.jl")
include("layers/deq.jl")
include("layers/sdeq.jl")
include("layers/mdeq.jl")
include("layers/smdeq.jl")
include("models.jl")
include("losses.jl")
include("zygote.jl")


export DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork
export MultiScaleDeepEquilibriumNetworkS4,
    MultiScaleSkipDeepEquilibriumNetworkS4
export DEQChain, MDEQChain
export get_and_clear_nfe!
export SupervisedLossContainer
export BroydenCache, broyden
export LinSolveKrylovJL
export SingleResolutionFeatures, MultiResolutionFeatures

end
