using FastDEQ
using DiffEqOperators
using DiffEqSensitivity
using Flux
using LinearAlgebra
using OrdinaryDiffEq
using SteadyStateDiffEq
using Test

@testset "FastDEQ.jl" begin
    # JVP with LinSolve
    mat = rand(5, 5) |> gpu
    x = rand(5, 1) |> gpu
    A = VecJacOperator((u, p, t) -> mat * u, x; autodiff = true)
    b = rand(5) |> gpu
    linsolve = LinSolveKrylovJL()
    @test A * vec(linsolve(zero(x), A, b)) ≈ b

    # Testing LinSolve with DiffEqSensitivity
    sensealg = SteadyStateAdjoint(
        autodiff = false,
        autojacvec = ZygoteVJP(),
        linsolve = LinSolveKrylovJL(),
    )
    model = DeepEquilibriumNetwork(
        Parallel(+, Dense(2, 2), Dense(2, 2)) |> gpu,
        DynamicSS(Tsit5(); abstol = 1f-1, reltol = 1f-1),
    )
    x = rand(2, 1) |> gpu
    ps = Flux.params(model)
    gs = Flux.gradient(() -> sum(model(x)), ps)
    for _p in ps
        @test all(isfinite.(gs[_p]))
    end
end
