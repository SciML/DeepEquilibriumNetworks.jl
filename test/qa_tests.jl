@testitem "Aqua" begin
    using Aqua

    Aqua.test_all(DeepEquilibriumNetworks; ambiguities=false)
    Aqua.test_ambiguities(DeepEquilibriumNetworks; recursive=false)
end

@testitem "ExplicitImports" begin
    import SciMLSensitivity, Zygote

    using ExplicitImports

    # Skip our own packages
    @test check_no_implicit_imports(DeepEquilibriumNetworks) === nothing
    ## AbstractRNG seems to be a spurious detection in LuxFluxExt
    @test check_no_stale_explicit_imports(DeepEquilibriumNetworks) === nothing
end
