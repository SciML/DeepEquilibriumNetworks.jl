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
    model = Chain(
        Dense(2, 2),
        DeepEquilibriumNetwork(
            Parallel(+, Dense(2, 2), Dense(2, 2)) |> gpu,
            DynamicSS(Tsit5(); abstol = 0.1f0, reltol = 0.1f0),
            sensealg = SteadyStateAdjoint(
                autodiff = false,
                autojacvec = ZygoteVJP(),
                linsolve = LinSolveKrylovJL(atol = 0.1f0, rtol = 0.1f0),
            )
        )
    ) |> gpu
    x = rand(Float32, 2, 1) |> gpu
    y = rand(Float32, 2, 1) |> gpu
    ps = Flux.params(model)
    gs = Flux.gradient(() -> sum(model(x) .- y), ps)
    gs.grads
    for _p in ps
        @test all(isfinite.(gs[_p]))
    end
end
