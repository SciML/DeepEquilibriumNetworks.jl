using SafeTestsets, Test, TestSetExtensions

@testset ExtendedTestSet "Deep Equilibrium Networks" begin
    @safetestset "Quality Assurance" include("qa.jl")
    @safetestset "Utilities" include("utils.jl")
    @safetestset "Layers" include("layers.jl")
end
