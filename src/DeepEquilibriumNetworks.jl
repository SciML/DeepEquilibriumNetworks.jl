module DeepEquilibriumNetworks

using DiffEqBase, LinearAlgebra, LinearSolve, Lux, OrdinaryDiffEq, Random, SciMLBase,
      SciMLOperators, SciMLSensitivity, Setfield, SimpleNonlinearSolve, Static,
      SteadyStateDiffEq

using DiffEqBase: AbstractSteadyStateProblem
using SciMLBase: AbstractNonlinearSolution, AbstractSteadyStateAlgorithm
using SimpleNonlinearSolve: AbstractSimpleNonlinearSolveAlgorithm
using TruncatedStacktraces: @truncate_stacktrace

import ChainRulesCore as CRC

const DEQs = DeepEquilibriumNetworks

include("solve.jl")
include("utils.jl")

include("layers/core.jl")
# include("layers/jacobian_stabilization.jl")
include("layers/deq.jl")
# include("layers/mdeq.jl")
# include("layers/neuralode.jl")

# include("adjoint.jl")

# Useful Shorthand
export DEQs

# DEQ Solvers
export ContinuousDEQSolver, DiscreteDEQSolver

# Utils
export EquilibriumSolution, DeepEquilibriumSolution # , estimate_jacobian_trace

# Networks
export DeepEquilibriumNetwork # , SkipDeepEquilibriumNetwork
# export MultiScaleDeepEquilibriumNetwork, MultiScaleSkipDeepEquilibriumNetwork

end
