module DeepEquilibriumNetworks

using ADTypes,
    DiffEqBase, LinearAlgebra, Lux, Random, SciMLBase, Statistics, SteadyStateDiffEq

import ChainRulesCore as CRC
import ConcreteStructs: @concrete
import ConstructionBase: constructorof
import Lux: AbstractExplicitLayer, AbstractExplicitContainerLayer
import TruncatedStacktraces: @truncate_stacktrace

import SciMLBase: AbstractNonlinearAlgorithm,
    AbstractODEAlgorithm, _unwrap_val, NonlinearSolution

# Useful Constants
const DEQs = DeepEquilibriumNetworks

include("layers.jl")
include("utils.jl")

# Exports
export DEQs, DeepEquilibriumSolution, DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork,
    MultiScaleDeepEquilibriumNetwork, MultiScaleSkipDeepEquilibriumNetwork,
    MultiScaleNeuralODE

end
