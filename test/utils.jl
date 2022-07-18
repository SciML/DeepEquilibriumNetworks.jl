import DeepEquilibriumNetworks as DEQs
import LinearAlgebra
import Static
import Test

include("test_utils.jl")

function test_deq_adjoint()
  deq_adjoint = Test.@test_nowarn DEQs.DeepEquilibriumAdjoint(0.1f0, 0.1f0, 100)

  Test.@test DEQs._check_adjoint_mode(deq_adjoint, Val(:vanilla))
  Test.@test !DEQs._check_adjoint_mode(deq_adjoint, Val(:random))

  deq_adjoint = Test.@test_nowarn DEQs.DeepEquilibriumAdjoint(0.1f0, 0.1f0, 100; mode=:jfb)

  Test.@test DEQs._check_adjoint_mode(deq_adjoint, Val(:jfb))
  Test.@test !DEQs._check_adjoint_mode(deq_adjoint, Val(:random))
  Test.@test !DEQs._check_adjoint_mode(deq_adjoint, Val(:vanilla))
end

function test_split_and_reshape()
  x1 = ones(Float32, 4, 4)
  x2 = fill!(zeros(Float32, 2, 4), 0.5f0)
  x3 = zeros(Float32, 1, 4)

  x = vcat(x1, x2, x3)
  split_idxs = Static.static(cumsum((0, size(x1, 1), size(x2, 1), size(x3, 1))))
  shapes = Static.static((size(x1, 1), size(x2, 1), size(x3, 1)))
  x_split = DEQs.split_and_reshape(x, split_idxs, shapes)

  Test.@test x1 == x_split[1]
  Test.@test x2 == x_split[2]
  Test.@test x3 == x_split[3]

  Test.@inferred DEQs.split_and_reshape(x, split_idxs, shapes)
  run_JET_tests(DEQs.split_and_reshape, x, split_idxs, shapes)

  return nothing
end

function test_init_identity_matrix()
  x = zeros(Float32, 5, 5, 2)
  imat = DEQs.init_identity_matrix(x, 0.5f0)

  Test.@test all(LinearAlgebra.diag(imat) .== 0.5f0)
  return nothing
end

Test.@testset "DeepEquilibriumAdjoint" begin test_deq_adjoint() end
Test.@testset "split_and_reshape" begin test_split_and_reshape() end
Test.@testset "init identity matrix" begin test_init_identity_matrix() end
