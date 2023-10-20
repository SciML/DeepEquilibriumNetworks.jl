using DeepEquilibriumNetworks, LinearAlgebra
using Test

include("test_utils.jl")

function test_split_and_reshape()
    x1 = ones(Float32, 4, 4)
    x2 = fill!(zeros(Float32, 2, 4), 0.5f0)
    x3 = zeros(Float32, 1, 4)

    x = vcat(x1, x2, x3)
    split_idxs = Val(cumsum((0, size(x1, 1), size(x2, 1), size(x3, 1))))
    shapes = Val((size(x1, 1), size(x2, 1), size(x3, 1)))
    x_split = DEQs.split_and_reshape(x, split_idxs, shapes)

    @test x1 == x_split[1]
    @test x2 == x_split[2]
    @test x3 == x_split[3]

    @inferred DEQs.split_and_reshape(x, split_idxs, shapes)
    run_JET_tests(DEQs.split_and_reshape, x, split_idxs, shapes)

    return nothing
end

@testset "split_and_reshape" begin
    test_split_and_reshape()
end
