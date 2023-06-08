using DeepEquilibriumNetworks, LinearAlgebra, Static
using Test

include("test_utils.jl")

function test_split_and_reshape()
    x1 = ones(Float32, 4, 4)
    x2 = fill!(zeros(Float32, 2, 4), 0.5f0)
    x3 = zeros(Float32, 1, 4)

    x = vcat(x1, x2, x3)
    split_idxs = static(cumsum((0, size(x1, 1), size(x2, 1), size(x3, 1))))
    shapes = static((size(x1, 1), size(x2, 1), size(x3, 1)))
    x_split = DEQs.split_and_reshape(x, split_idxs, shapes)

    @test x1 == x_split[1]
    @test x2 == x_split[2]
    @test x3 == x_split[3]

    @inferred DEQs.split_and_reshape(x, split_idxs, shapes)
    run_JET_tests(DEQs.split_and_reshape, x, split_idxs, shapes)

    return nothing
end

function test_init_identity_matrix()
    x = zeros(Float32, 5, 5, 2)
    imat = DEQs.init_identity_matrix(x, 0.5f0)

    @test all(diag(imat) .== 0.5f0)
    return nothing
end

@testset "split_and_reshape" begin
    test_split_and_reshape()
end
@testset "init identity matrix" begin
    test_init_identity_matrix()
end
