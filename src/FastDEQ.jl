module FastDEQ

using ChainRulesCore,
    ComponentArrays,
    CUDA,
    DiffEqBase,
    DiffEqCallbacks,
    DiffEqSensitivity,
    Functors,
    LinearAlgebra,
    LinearSolve,
    Lux,
    MLUtils,
    OrdinaryDiffEq,
    SciMLBase,
    Setfield,
    Static,
    Statistics,
    SteadyStateDiffEq,
    UnPack,
    Zygote

import DiffEqSensitivity: AbstractAdjointSensitivityAlgorithm
import Lux: AbstractExplicitContainerLayer, initialparameters, initialstates, parameterlength, statelength
import Random: AbstractRNG

include("operator.jl")

include("solvers/continuous.jl")
include("solvers/discrete.jl")
include("solvers/termination.jl")

include("solve.jl")
include("utils.jl")

include("layers/core.jl")
include("layers/jacobian_stabilization.jl")
include("layers/deq.jl")
include("layers/mdeq.jl")
include("layers/chain.jl")

include("adjoint.jl")

# DEQ Solvers
export ContinuousDEQSolver, DiscreteDEQSolver, BroydenSolver, LimitedMemoryBroydenSolver

# Utils
export NormalInitializer, DeepEquilibriumAdjoint, compute_deq_jacobian_loss, DeepEquilibriumSolution

export DeepEquilibriumNetwork,
    SkipDeepEquilibriumNetwork, MultiScaleDeepEquilibriumNetwork, MultiScaleSkipDeepEquilibriumNetwork, DEQChain

end
