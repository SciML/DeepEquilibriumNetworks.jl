module FastDEQ

using CUDA
using DiffEqBase
using DiffEqSensitivity
using Flux
using Functors
using LinearAlgebra
using LinearSolve
using OrdinaryDiffEq
using SteadyStateDiffEq
using UnPack
using Zygote


abstract type AbstractDeepEquilibriumNetwork end

function construct_iterator end


include("utils.jl")
include("layers.jl")
include("solvers/broyden.jl")
include("solvers/linsolve.jl")


export DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork
export BroydenCache, broyden
export LinSolveKrylovJL, LinearScaledJacVecOperator
export parameter_destructure

end
