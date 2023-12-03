using DeepEquilibriumNetworks, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(DeepEquilibriumNetworks)
    Aqua.test_ambiguities(DeepEquilibriumNetworks, recursive = false, broken = true)
    Aqua.test_deps_compat(DeepEquilibriumNetworks)
    Aqua.test_piracies(DeepEquilibriumNetworks, broken = true)
    Aqua.test_project_extras(DeepEquilibriumNetworks)
    Aqua.test_stale_deps(DeepEquilibriumNetworks)
    Aqua.test_unbound_args(DeepEquilibriumNetworks, broken = true)
    Aqua.test_undefined_exports(DeepEquilibriumNetworks, broken = true)
end
