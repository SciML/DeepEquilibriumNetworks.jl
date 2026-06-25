using SciMLTesting, DeepEquilibriumNetworks, Test

run_qa(
    DeepEquilibriumNetworks;
    explicit_imports = true,
    # `Aqua.test_all`'s default ambiguities check recurses into deps and is noisy; run
    # it only against this package's own methods (the prior hand-rolled qa.jl disabled
    # the recursive sweep and called `test_ambiguities(...; recursive = false)`).
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    ei_kwargs = (;
        # Names accessed as `Mod.name` that are not (yet) public in their owning
        # package; tracked here until those packages mark them public on release.
        all_qualified_accesses_are_public = (;
            ignore = (
                :DEStats, :NLStats,      # SciMLBase
                :Fix1,                   # Base
                :apply, :initialstates, :replicate, :setup, :update_state,  # LuxCore
                :getproperty,            # Lux.LuxOps
            ),
        ),
        # Names brought in via `using Mod: name` that are not (yet) public in Mod.
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractNonlinearAlgorithm, :AbstractODEAlgorithm, :_unwrap_val,  # SciMLBase
                :solve,                  # CommonSolve
            ),
        ),
    ),
)

@testset "Doctests" begin
    using Documenter

    doctestexpr = quote
        using DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq, NonlinearSolve
    end

    DocMeta.setdocmeta!(DeepEquilibriumNetworks, :DocTestSetup, doctestexpr; recursive = true)
    doctest(DeepEquilibriumNetworks; manual = false)
end
