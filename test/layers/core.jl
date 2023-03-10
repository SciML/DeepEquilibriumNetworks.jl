using DeepEquilibriumNetworks, SciMLBase
using Test

include("../test_utils.jl")

function test_check_unrolled_mode()
  @test SciMLBase._unwrap_val(DEQs._check_unrolled_mode(Val(10)))
  @test !SciMLBase._unwrap_val(DEQs._check_unrolled_mode(Val(0)))
  @test SciMLBase._unwrap_val(DEQs._check_unrolled_mode((; fixed_depth=Val(10))))
  @test !SciMLBase._unwrap_val(DEQs._check_unrolled_mode((; fixed_depth=Val(0))))
  return nothing
end

function test_get_unrolled_depth()
  @test DEQs._get_unrolled_depth(Val(10)) == 10
  @test DEQs._get_unrolled_depth((; fixed_depth=Val(10))) == 10
  return nothing
end

function test_deep_equilibrium_solution()
  sol = @test_nowarn DEQs.DeepEquilibriumSolution(randn(10), randn(10), randn(10), 0.4, 10)
  @test_nowarn println(sol)
  return nothing
end

@testset "check unrolled mode" begin test_check_unrolled_mode() end
@testset "get unrolled depth" begin test_get_unrolled_depth() end
@testset "deep equilibrium solution" begin test_deep_equilibrium_solution() end
