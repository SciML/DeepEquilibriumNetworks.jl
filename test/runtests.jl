using FastDEQ
using CUDA
using DiffEqOperators
using DiffEqSensitivity
using Flux
using LinearAlgebra
using OrdinaryDiffEq
using Random
using SteadyStateDiffEq
using Test

@testset "FastDEQ.jl" begin
    # JVP with LinSolve
    Random.seed!(0)

    mat = rand(5, 5) |> gpu
    x = rand(5, 1) |> gpu
    A = VecJacOperator((u, p, t) -> mat * u, x; autodiff = true)
    b = rand(5) |> gpu
    linsolve = LinSolveKrylovJL()
    @test A * vec(linsolve(zero(x), A, b)) ≈ b

    # Testing LinSolve with DiffEqSensitivity
    Random.seed!(0)

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
    for _p in ps
        @test all(isfinite.(gs[_p]))
    end

    # Testing SkipDEQ
    Random.seed!(0)

    model = Chain(
        Dense(2, 2),
        SkipDeepEquilibriumNetwork(
            Parallel(+, Dense(2, 2), Dense(2, 2)) |> gpu,
            Dense(2, 2) |> gpu,
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
    gs = Flux.gradient(() -> begin
        ŷ, z = model(x)
        sum(abs2, ŷ .- y) + sum(abs2, ŷ .- z)
    end, ps)
    for _p in ps
        @test all(isfinite.(gs[_p]))
    end

    # Testing MultiScaleDEQ
    Random.seed!(0)

    CUDA.allowscalar() do

    model = MultiScaleDeepEquilibriumNetworkS4(
        (
            Parallel(+, Dense(4, 4, tanh), Dense(4, 4, tanh)),
            Dense(3, 3, tanh),
            Dense(2, 2, tanh),
            Dense(1, 1, tanh),
        ) .|> gpu,
        (
            (identity, Dense(4, 3, tanh), Dense(4, 2, tanh), Dense(4, 1, tanh)) .|> gpu,
            (Dense(3, 4, tanh), identity, Dense(3, 2, tanh), Dense(3, 1, tanh)) .|> gpu,
            (Dense(2, 4, tanh), Dense(2, 3, tanh), identity, Dense(2, 1, tanh)) .|> gpu,
            (Dense(1, 4, tanh), Dense(1, 3, tanh), Dense(1, 2, tanh), identity) .|> gpu,
        ),
        DynamicSS(Tsit5(); abstol = 0.1, reltol = 0.1),
        sensealg = SteadyStateAdjoint(
            autodiff = false,
            autojacvec = ZygoteVJP(),
            linsolve = LinSolveKrylovJL(atol = 0.1, rtol = 0.1),
        ),
        maxiter = 100,
    )
    x = rand(4, 128) |> gpu
    sol = model(x)
    ps = Flux.params(model)
    gs = Flux.gradient(
        () -> begin
            x1, x2, x3, x4 = model(x)
            return (
                sum(abs2, x1 .- x) +
                sum(abs2, x2) +
                sum(abs2, x3) +
                sum(abs2, x4)
            )
        end,
        ps
    )
    for _p in ps
        @test all(isfinite.(gs[_p]))
    end

    end

    # Testing MultiScaleSkipDEQ
    Random.seed!(0)

    CUDA.allowscalar() do

    model = MultiScaleSkipDeepEquilibriumNetworkS4(
        (
            Parallel(+, Dense(4, 4, tanh), Dense(4, 4, tanh)),
            Dense(3, 3, tanh),
            Dense(2, 2, tanh),
            Dense(1, 1, tanh),
        ) .|> gpu,
        (
            (identity, Dense(4, 3, tanh), Dense(4, 2, tanh), Dense(4, 1, tanh)) .|> gpu,
            (Dense(3, 4, tanh), identity, Dense(3, 2, tanh), Dense(3, 1, tanh)) .|> gpu,
            (Dense(2, 4, tanh), Dense(2, 3, tanh), identity, Dense(2, 1, tanh)) .|> gpu,
            (Dense(1, 4, tanh), Dense(1, 3, tanh), Dense(1, 2, tanh), identity) .|> gpu,
        ),
        (
            Dense(4, 4, tanh),
            Dense(4, 3, tanh),
            Dense(4, 2, tanh),
            Dense(4, 1, tanh),
        ) .|> gpu,
        DynamicSS(Tsit5(); abstol = 0.1, reltol = 0.1),
        sensealg = SteadyStateAdjoint(
            autodiff = false,
            autojacvec = ZygoteVJP(),
            linsolve = LinSolveKrylovJL(atol = 0.1, rtol = 0.1),
        ),
        maxiter = 100,
    )
    x = rand(4, 128) |> gpu
    sol = model(x)
    ps = Flux.params(model)
    Flux.gradient(
        () -> begin
            (x1, x2, x3, x4), (x1g, x2g, x3g, x4g) = model(x)
            return (
                sum(abs2, x1 .- x) +
                sum(abs2, x2) +
                sum(abs2, x3) +
                sum(abs2, x4) +
                sum(abs2, x1 .- x1g) +
                sum(abs2, x2 .- x2g) +
                sum(abs2, x3 .- x3g) +
                sum(abs2, x4 .- x4g)
            )
        end,
        ps
    )
    for _p in ps
        @test all(isfinite.(gs[_p]))
    end

    end
end
