using SafeTestsets, Test

@testset "Deep Equilibrium Networks" begin
    @safetestset "Quality Assurance" begin
        include("qa.jl")
    end
    @safetestset "Utilities" begin
        include("utils.jl")
    end
    @safetestset "Layers" begin
        include("layers.jl")
    end
end
