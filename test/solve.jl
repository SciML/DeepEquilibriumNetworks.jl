using DeepEquilibriumNetworks, SciMLBase, SteadyStateDiffEq
using Test

simple_dudt(u, p, t) = 0.9f0 .* u .- u

function test_continuous_deq_solver()
    prob = SteadyStateProblem(simple_dudt, [1.0f0], SciMLBase.NullParameters())

    sol = solve(prob, ContinuousDEQSolver(); save_everystep=true)

    @test sol isa DEQs.EquilibriumSolution
    @test abs(sol.u[1]) <= 1.0f-4

    return nothing
end

function test_discrete_deq_solver()
    prob = SteadyStateProblem(simple_dudt, reshape([1.0f0], 1, 1),
        SciMLBase.NullParameters())

    sol = solve(prob, DiscreteDEQSolver())

    @test sol isa DEQs.EquilibriumSolution
    @test abs(sol.u[1]) <= 1.0f-4

    return nothing
end

Test.@testset "Continuous Steady State Solve" begin
    test_continuous_deq_solver()
end
Test.@testset "Discrete Steady State Solve" begin
    test_discrete_deq_solver()
end
