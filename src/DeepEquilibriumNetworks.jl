module DeepEquilibriumNetworks

using ADTypes: AutoFiniteDiff, AutoForwardDiff, AutoZygote
using ChainRulesCore: ChainRulesCore
using CommonSolve: solve
using ConcreteStructs: @concrete
using ConstructionBase: ConstructionBase
using DiffEqBase: DiffEqBase, AbsNormTerminationMode
using FastClosures: @closure
using Lux: Lux, BranchLayer, Chain, NoOpLayer, Parallel, RepeatedLayer, StatefulLuxLayer,
           WrappedFunction
using LuxCore: LuxCore, AbstractExplicitLayer, AbstractExplicitContainerLayer
using NNlib: ⊠
using Random: Random, AbstractRNG, randn!
using SciMLBase: SciMLBase, AbstractNonlinearAlgorithm, AbstractODEAlgorithm,
                 NonlinearSolution, ODESolution, ODEFunction, ODEProblem,
                 SteadyStateProblem, _unwrap_val
using SteadyStateDiffEq: DynamicSS, SSRootfind

# Useful Constants
const CRC = ChainRulesCore
const DEQs = DeepEquilibriumNetworks

include("layers.jl")
include("utils.jl")

# Exports
export DEQs, DeepEquilibriumSolution, DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork,
       MultiScaleDeepEquilibriumNetwork, MultiScaleSkipDeepEquilibriumNetwork,
       MultiScaleNeuralODE

end
