module DeepEquilibriumNetworks

import Reexport: @reexport

@reexport using Lux, NonlinearSolve, OrdinaryDiffEq, SciMLSensitivity

using DiffEqBase,
    LinearAlgebra, LinearSolve, Random, SciMLBase, Statistics,
    SteadyStateDiffEq, Zygote

import DiffEqBase: AbstractSteadyStateProblem
import SciMLBase: AbstractNonlinearSolution, AbstractSteadyStateAlgorithm
import NonlinearSolve: AbstractNonlinearSolveAlgorithm
import TruncatedStacktraces: @truncate_stacktrace

import ChainRulesCore as CRC
import ConcreteStructs: @concrete

const DEQs = DeepEquilibriumNetworks
const ∂∅ = CRC.NoTangent()

## FIXME: Uses of nothing was removed in Lux 0.5 with a deprecation. It was not updated
##        here
Lux.parameterlength(::Nothing) = 0
Lux.statelength(::Nothing) = 0

include("solve.jl")
include("utils.jl")

include("layers/core.jl")
include("layers/jacobian_stabilization.jl")
include("layers/deq.jl")
include("layers/mdeq.jl")
include("layers/evaluate.jl")

include("chainrules.jl")

# Useful Shorthand
export DEQs

# DEQ Solvers
export ContinuousDEQSolver, DiscreteDEQSolver

# Utils
export EquilibriumSolution, DeepEquilibriumSolution, estimate_jacobian_trace

# Networks
export DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork, MultiScaleInputLayer,
    MultiScaleNeuralODE, MultiScaleDeepEquilibriumNetwork,
    MultiScaleSkipDeepEquilibriumNetwork

end
