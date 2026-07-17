using SciMLTesting, DeepEquilibriumNetworks, Test

run_qa(
    DeepEquilibriumNetworks;
    explicit_imports = true,
    # `DEQs` is a module-valued alias documented in prose on the API page. Documenter
    # canonicalizes its `@docs` entry to the module binding, so it cannot be represented
    # as a rendered alias binding.
    api_docs_kwargs = (; rendered = true, rendered_ignore = (:DEQs,)),
    # `Aqua.test_all`'s default ambiguities check recurses into deps and is noisy; run
    # it only against this package's own methods (the prior hand-rolled qa.jl disabled
    # the recursive sweep and called `test_ambiguities(...; recursive = false)`).
    aqua_kwargs = (; ambiguities = (; recursive = false)),
)

@testset "Doctests" begin
    using Documenter

    doctestexpr = quote
        using DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq, NonlinearSolve
    end

    DocMeta.setdocmeta!(DeepEquilibriumNetworks, :DocTestSetup, doctestexpr; recursive = true)
    doctest(DeepEquilibriumNetworks; manual = false)
end
