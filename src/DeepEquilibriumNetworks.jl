module DeepEquilibriumNetworks

using DiffEqBase, LinearAlgebra, OrdinaryDiffEq, SciMLBase, SciMLOperators,
      SimpleNonlinearSolve, SteadyStateDiffEq
using DiffEqBase: AbstractSteadyStateProblem
using SciMLBase: AbstractNonlinearSolution
using SteadyStateDiffEq: SteadyStateDiffEqAlgorithm, _get_termination_condition

const DEQs = DeepEquilibriumNetworks

include("solvers/solvers.jl")
include("solvers/discrete/broyden.jl")
include("solvers/discrete/limited_memory_broyden.jl")

include("solve.jl")
include("utils.jl")

include("layers/core.jl")
include("layers/jacobian_stabilization.jl")
include("layers/deq.jl")
include("layers/mdeq.jl")
include("layers/neuralode.jl")

include("adjoint.jl")

# Useful Shorthand
export DEQs

# DEQ Solvers
export ContinuousDEQSolver, DiscreteDEQSolver, BroydenSolver, LimitedMemoryBroydenSolver

# Utils
export DeepEquilibriumAdjoint, DeepEquilibriumSolution, estimate_jacobian_trace

# Networks
export DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork
export MultiScaleDeepEquilibriumNetwork, MultiScaleSkipDeepEquilibriumNetwork

end
