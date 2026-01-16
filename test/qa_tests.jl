const QA_GROUP = lowercase(get(ENV, "BACKEND_GROUP", get(ENV, "GROUP", "all")))

@testitem "Aqua" tags=[:qa] skip=(QA_GROUP != "qa") begin
    using Aqua

    Aqua.test_all(DeepEquilibriumNetworks; ambiguities = false)
    Aqua.test_ambiguities(DeepEquilibriumNetworks; recursive = false)
end

@testitem "ExplicitImports" tags=[:qa] skip=(QA_GROUP != "qa") begin
    import SciMLSensitivity

    using ExplicitImports

    @test check_no_implicit_imports(DeepEquilibriumNetworks) === nothing
    @test check_no_stale_explicit_imports(DeepEquilibriumNetworks) === nothing
    @test check_all_qualified_accesses_via_owners(DeepEquilibriumNetworks) === nothing
end

@testitem "Doctests" tags=[:qa] skip=(QA_GROUP != "qa") begin
    using Documenter

    doctestexpr = quote
        using DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq, NonlinearSolve
    end

    DocMeta.setdocmeta!(DeepEquilibriumNetworks, :DocTestSetup, doctestexpr; recursive = true)
    doctest(DeepEquilibriumNetworks; manual = false)
end
