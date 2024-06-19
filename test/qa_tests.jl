@testitem "Aqua" begin
    using Aqua

    Aqua.test_all(DeepEquilibriumNetworks; ambiguities=false)
    Aqua.test_ambiguities(DeepEquilibriumNetworks; recursive=false)
end

@testitem "ExplicitImports" begin
    import SciMLSensitivity

    using ExplicitImports

    @test check_no_implicit_imports(DeepEquilibriumNetworks) === nothing
    @test check_no_stale_explicit_imports(DeepEquilibriumNetworks) === nothing
    @test check_all_qualified_accesses_via_owners(DeepEquilibriumNetworks) === nothing
end

@testitem "Doctests" begin
    using Documenter

    doctestexpr = quote
        using DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq, NonlinearSolve
    end

    DocMeta.setdocmeta!(DeepEquilibriumNetworks, :DocTestSetup, doctestexpr; recursive=true)
    doctest(DeepEquilibriumNetworks; manual=false)
end
