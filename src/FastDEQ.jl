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
using SciMLBase
using Statistics
using SteadyStateDiffEq
using UnPack
using Zygote


abstract type AbstractDeepEquilibriumNetwork end

function construct_iterator end


include("utils.jl")
include("solvers/broyden.jl")
include("solvers/linsolve.jl")
include("layers.jl")
include("models.jl")
include("losses.jl")


export DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork
export MultiScaleDeepEquilibriumNetwork
export DEQChain, MDEQChain
export get_and_clear_nfe!
export SupervisedLossContainer
export BroydenCache, broyden
export LinSolveKrylovJL, LinearScaledJacVecOperator, VecJacOperator
export parameter_destructure

end
