using FastDEQ
using DiffEqOperators
using DiffEqSensitivity
using Flux
using LinearAlgebra
using Test

@testset "FastDEQ.jl" begin
    # JVP with LinSolve
    model = Chain(Dense(5, 1), Dense(1, 5)) |> gpu
    x = rand(5, 1) |> gpu
    p, re = Flux.destructure(model)
    A = LinearScaledJacVecOperator(JacVecOperator((z, u, p, t) -> z .= re(p)(u), x, p; autodiff = true),
                                   Diagonal(ones(length(x)) |> gpu))
    b = rand(5) |> gpu

    linsolve = LinSolveKrylovJL()
    res = zero(vec(x))
    @test mul!(zero(b), A, vec(linsolve(zero(x), A, b))) ≈ b


    # Testing LinSolve with DiffEqSensitivity
    # sensealg = SteadyStateAdjoint(autodiff = false, autojacvec = ZygoteVJP(), linsolve = LinSolveKrylovJL())

end
