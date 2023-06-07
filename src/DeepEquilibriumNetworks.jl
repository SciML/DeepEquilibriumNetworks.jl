module DeepEquilibriumNetworks

using CUDA,
  DiffEqBase,
  LinearAlgebra,
  LinearSolve,
  Lux,
  MLUtils,
  OrdinaryDiffEq,
  Random,
  SciMLBase,
  SciMLSensitivity,
  Setfield,
  SimpleNonlinearSolve,
  Static,
  Statistics,
  SteadyStateDiffEq,
  Zygote,
  ZygoteRules

using DiffEqBase: AbstractSteadyStateProblem
using SciMLBase: AbstractNonlinearSolution, AbstractSteadyStateAlgorithm
using SimpleNonlinearSolve: AbstractSimpleNonlinearSolveAlgorithm
using TruncatedStacktraces: @truncate_stacktrace

import ChainRulesCore as CRC

const DEQs = DeepEquilibriumNetworks

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
export DeepEquilibriumNetwork,
  SkipDeepEquilibriumNetwork,
  MultiScaleInputLayer,
  MultiScaleNeuralODE,
  MultiScaleDeepEquilibriumNetwork,
  MultiScaleSkipDeepEquilibriumNetwork

end
