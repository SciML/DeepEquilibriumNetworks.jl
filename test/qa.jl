using DeepEquilibriumNetworks, Aqua, Test
import ChainRulesCore as CRC

@testset "Aqua" begin
    Aqua.test_all(DeepEquilibriumNetworks; ambiguities=false)
    Aqua.test_ambiguities(DeepEquilibriumNetworks; recursive=false)
end
