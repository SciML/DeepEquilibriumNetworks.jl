module DeepEquilibriumNetworksSciMLSensitivityExt

# Linear Solve is a dependency of SciMLSensitivity, so we only need to load SciMLSensitivity
# to load this extension
using LinearSolve: SimpleGMRES
using SciMLBase: SteadyStateProblem, ODEProblem
using SciMLSensitivity: SteadyStateAdjoint, GaussAdjoint, ZygoteVJP
using DeepEquilibriumNetworks: DEQs

@inline function DEQs.__default_sensealg(prob::SteadyStateProblem)
    # We want to avoid the cost for cache construction for linsolve = nothing
    # For small problems we should use concrete jacobian but we assume users want to solve
    # large problems with this package so we default to GMRES and avoid runtime dispatches
    linsolve = SimpleGMRES{true}(; blocksize=prod(size(prob.u0)[1:(end - 1)]))
    linsolve_kwargs = (; maxiters=10, abstol=1e-3, reltol=1e-3)
    return SteadyStateAdjoint(; linsolve, linsolve_kwargs, autojacvec=ZygoteVJP())
end
@inline DEQs.__default_sensealg(::ODEProblem) = GaussAdjoint(; autojacvec=ZygoteVJP())

end
