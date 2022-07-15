import DeepEquilibriumNetworks as DEQs
import LinearAlgebra
import Test

f(x) = x .^ 2 .- 2 .* x .+ 1

function test_limited_memory_broyden_convergence()
  solver = DEQs.LimitedMemoryBroydenSolver()
  x0 = [-2.0f0 2.0f0]
  terminate_stats = Dict{Symbol, Any}(:best_objective_value => real(eltype(x0))(Inf),
                                      :best_objective_value_iteration => nothing)
  terminate_condition = DEQs.get_terminate_condition(DEQs.DiscreteDEQSolver(solver),
                                                     terminate_stats)
  maxiters = 150

  xs, _ = DEQs.nlsolve(solver, f, x0; terminate_condition, maxiters)

  Test.@test LinearAlgebra.norm(f(xs[end])) <= 0.01

  return nothing
end

Test.@testset "Limited Memory Broyden" begin test_limited_memory_broyden_convergence() end
