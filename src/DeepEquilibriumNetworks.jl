module DeepEquilibriumNetworks

using ADTypes: AutoFiniteDiff, AutoForwardDiff, AutoZygote
using ChainRulesCore: ChainRulesCore
using CommonSolve: solve, init
using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase
using NonlinearSolveBase: AbsNormTerminationMode
using FastClosures: @closure
using Random: Random, AbstractRNG, randn!
using SciMLBase: SciMLBase, AbstractNonlinearAlgorithm, AbstractODEAlgorithm,
    NonlinearSolution, ODESolution, ODEFunction, ODEProblem,
    SteadyStateProblem, _unwrap_val
using SciMLSensitivity: SteadyStateAdjoint, GaussAdjoint, ZygoteVJP
using Static: StaticSymbol, StaticInt, known, static

using Lux: Lux, LuxOps, BranchLayer, Chain, NoOpLayer, Parallel, RepeatedLayer,
    StatefulLuxLayer, WrappedFunction
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer
using NNlib: ⊠
using SteadyStateDiffEq: DynamicSS, SSRootfind

# Useful Constants
const CRC = ChainRulesCore
const DEQs = DeepEquilibriumNetworks

include("layers.jl")
include("utils.jl")
include("precompilation.jl")

# Exports
export DEQs, DeepEquilibriumSolution, DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork,
    MultiScaleDeepEquilibriumNetwork, MultiScaleSkipDeepEquilibriumNetwork,
    MultiScaleNeuralODE

end
