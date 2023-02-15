using DeepEquilibriumNetworks, OrdinaryDiffEq, SciMLBase, SteadyStateDiffEq

simple_dudt(u, p, t) = 0.9f0 .* u .- u

function test_continuous_deq_solver()
  prob = SteadyStateProblem(ODEFunction(simple_dudt), [1.0f0], SciMLBase.NullParameters())

  sol = solve(prob,
              ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0,
                                  mode=SteadyStateTerminationMode.AbsNorm))

  Test.@test sol isa DEQs.EquilibriumSolution
  Test.@test abs(sol.u[1]) <= 1.0f-2

  return nothing
end

function test_discrete_deq_solver()
  prob = SteadyStateProblem(ODEFunction(simple_dudt), [1.0f0], SciMLBase.NullParameters())

  sol = SciMLBase.solve(prob,
                        DiscreteDEQSolver(BroydenSolver(); abstol_termination=0.01f0,
                                          reltol_termination=0.01f0,
                                          mode=SteadyStateTerminationMode.AbsNorm))

  Test.@test sol isa DEQs.EquilibriumSolution
  Test.@test abs(sol.u[1]) <= 1.0f-2

  return nothing
end

Test.@testset "Continuous Steady State Solve" begin test_continuous_deq_solver() end
Test.@testset "Discrete Steady State Solve" begin test_discrete_deq_solver() end
