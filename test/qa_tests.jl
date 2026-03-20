using DeepEquilibriumNetworks, Test

@testset "Aqua" begin
    using Aqua

    # treat_as_own: Adjoint convert method is a workaround for LinearSolve.jl bug
    # (missing convert(Adjoint{T,S}, ::S) in LinearAlgebra)
    using LinearAlgebra: Adjoint
    Aqua.test_all(
        DeepEquilibriumNetworks;
        ambiguities = false, piracies = (; treat_as_own = [Adjoint])
    )
    Aqua.test_ambiguities(DeepEquilibriumNetworks; recursive = false)
end

@testset "ExplicitImports" begin
    import SciMLSensitivity

    using ExplicitImports

    @test check_no_implicit_imports(DeepEquilibriumNetworks) === nothing
    @test check_no_stale_explicit_imports(DeepEquilibriumNetworks) === nothing
    @test check_all_qualified_accesses_via_owners(DeepEquilibriumNetworks) === nothing
end

@testset "Doctests" begin
    using Documenter

    doctestexpr = quote
        using DeepEquilibriumNetworks, Lux, Random, OrdinaryDiffEq, NonlinearSolve
    end

    DocMeta.setdocmeta!(DeepEquilibriumNetworks, :DocTestSetup, doctestexpr; recursive = true)
    doctest(DeepEquilibriumNetworks; manual = false)
end
