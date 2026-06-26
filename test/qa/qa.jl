using SciMLTesting, DeepEquilibriumNetworks, Test

run_qa(
    DeepEquilibriumNetworks;
    explicit_imports = true,
    # `Aqua.test_all`'s default ambiguities check recurses into deps and is noisy; run
    # it only against this package's own methods (the prior hand-rolled qa.jl disabled
    # the recursive sweep and called `test_ambiguities(...; recursive = false)`).
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    ei_kwargs = (;
        # SciMLBase internals not yet marked public; tracked until SciMLBase
        # exports/declares them public on a future release.
        all_qualified_accesses_are_public = (; ignore = (:DEStats,)),  # SciMLBase
        all_explicit_imports_are_public = (;
            ignore = (:AbstractNonlinearAlgorithm, :_unwrap_val),  # SciMLBase
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
