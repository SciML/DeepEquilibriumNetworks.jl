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

## FIXME: Remove once Manifest is removed
using SciMLBase, SciMLSensitivity

@inline __default_sensealg(::SteadyStateProblem) = SteadyStateAdjoint(;
    autojacvec=ZygoteVJP(), linsolve_kwargs=(; maxiters=10, abstol=1e-3, reltol=1e-3))
@inline __default_sensealg(::ODEProblem) = GaussAdjoint(; autojacvec=ZygoteVJP())
## FIXME: Remove once Manifest is removed

# Exports
export DEQs, DeepEquilibriumSolution, DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork,
    MultiScaleDeepEquilibriumNetwork, MultiScaleSkipDeepEquilibriumNetwork,
    MultiScaleNeuralODE

end
