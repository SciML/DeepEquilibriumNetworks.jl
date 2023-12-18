module DeepEquilibriumNetworksSciMLSensitivityExt

using SciMLBase, SciMLSensitivity
import DeepEquilibriumNetworks: __default_sensealg

@inline __default_sensealg(::SteadyStateProblem) = SteadyStateAdjoint(;
    autojacvec=ZygoteVJP())
@inline __default_sensealg(::ODEProblem) = GaussAdjoint(; autojacvec=ZygoteVJP())

end
