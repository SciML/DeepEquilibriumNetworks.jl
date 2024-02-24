module DeepEquilibriumNetworks

import PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ADTypes, DiffEqBase, FastClosures, LinearAlgebra, Lux, Random, SciMLBase,
          Statistics, SteadyStateDiffEq

    import ChainRulesCore as CRC
    import ConcreteStructs: @concrete
    import ConstructionBase: constructorof
    import Lux: AbstractExplicitLayer, AbstractExplicitContainerLayer
    import SciMLBase: AbstractNonlinearAlgorithm,
                      AbstractODEAlgorithm, _unwrap_val, NonlinearSolution
    import TruncatedStacktraces: @truncate_stacktrace
end

# Useful Constants
const DEQs = DeepEquilibriumNetworks

include("layers.jl")
include("utils.jl")

# Exports
export DEQs, DeepEquilibriumSolution, DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork,
       MultiScaleDeepEquilibriumNetwork, MultiScaleSkipDeepEquilibriumNetwork,
       MultiScaleNeuralODE

end
