using DeepEquilibriumNetworks, SciMLBase, SteadyStateDiffEq
using Test

simple_dudt(u, p, t) = 0.9f0 .* u .- u

function test_continuous_deq_solver()
    prob = SteadyStateProblem(simple_dudt, [1.0f0])

    sol = solve(prob, ContinuousDEQSolver(); save_everystep=true)

    @test sol isa DEQs.EquilibriumSolution
    @test abs(sol.u[1]) ≤ 1.0f-4

    return nothing
end

function test_discrete_deq_solver(; solver=nothing)
    prob = SteadyStateProblem(simple_dudt, reshape([1.0f0], 1, 1))

    sol = solve(prob, solver === nothing ? DiscreteDEQSolver() : DiscreteDEQSolver(solver))

    @test sol isa DEQs.EquilibriumSolution
    @test abs(sol.u[1]) ≤ 1.0f-4

    return nothing
end

@testset "Continuous Steady State Solve" begin
    test_continuous_deq_solver()
end
@testset "Discrete Steady State Solve" begin
    test_discrete_deq_solver(; solver=nothing)  # Default
    test_discrete_deq_solver(; solver=NewtonRaphson())
    test_discrete_deq_solver(; solver=LevenbergMarquardt())
end
