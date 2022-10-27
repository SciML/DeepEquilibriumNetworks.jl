import DeepEquilibriumNetworks as DEQs
import Test

include("../test_utils.jl")

function test_check_unrolled_mode()
  Test.@test DEQs._check_unrolled_mode(Val(10))
  Test.@test !DEQs._check_unrolled_mode(Val(0))
  Test.@test DEQs._check_unrolled_mode((; fixed_depth=Val(10)))
  Test.@test !DEQs._check_unrolled_mode((; fixed_depth=Val(0)))
end

function test_get_unrolled_depth()
  Test.@test DEQs._get_unrolled_depth(Val(10)) == 10
  Test.@test DEQs._get_unrolled_depth((; fixed_depth=Val(10))) == 10
end

Test.@testset "check unrolled mode" begin test_check_unrolled_mode() end
Test.@testset "get unrolled depth" begin test_get_unrolled_depth() end
