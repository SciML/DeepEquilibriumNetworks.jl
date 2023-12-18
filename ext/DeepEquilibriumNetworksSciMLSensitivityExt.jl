module DeepEquilibriumNetworksSciMLSensitivityExt

using SciMLBase, SciMLSensitivity
import DeepEquilibriumNetworks: __default_sensealg

@inline __default_sensealg(::SteadyStateProblem) = SteadyStateAdjoint(;
    autojacvec=ZygoteVJP(), linsolve_kwargs=(; maxiters=10, abstol=1e-3, reltol=1e-3))
@inline __default_sensealg(::ODEProblem) = GaussAdjoint(; autojacvec=ZygoteVJP())

end
