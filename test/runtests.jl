using FastDEQ
using CUDA
using Flux
using LinearAlgebra
using Random
using Test

@testset "FastDEQ.jl" begin
    mse_loss_function = SupervisedLossContainer(Flux.Losses.mse, 1.0f0)

    # Testing DEQ
    Random.seed!(0)

    model = gpu(DEQChain(Dense(2, 2),
                         DeepEquilibriumNetwork(Parallel(+, Dense(2, 2; bias=false), Dense(2, 2; bias=false)),
                                                get_default_dynamicss_solver(0.1f0, 0.1f0);
                                                sensealg=get_default_ssadjoint(0.1f0, 0.1f0, 10))))
    x = gpu(rand(Float32, 2, 1))
    y = gpu(rand(Float32, 2, 1))
    ps = Flux.params(model)
    gs = Flux.gradient(() -> mse_loss_function(model, x, y), ps)
    for _p in ps
        @test all(isfinite.(gs[_p]))
    end

    # Testing SkipDEQ
    Random.seed!(0)

    model = gpu(DEQChain(Dense(2, 2),
                         SkipDeepEquilibriumNetwork(Parallel(+, Dense(2, 2), Dense(2, 2)), Dense(2, 2),
                                                    get_default_dynamicss_solver(0.1f0, 0.1f0);
                                                    sensealg=get_default_ssadjoint(0.1f0, 0.1f0, 10))))
    x = gpu(rand(Float32, 2, 1))
    y = gpu(rand(Float32, 2, 1))
    ps = Flux.params(model)
    gs = Flux.gradient(() -> mse_loss_function(model, x, y), ps)
    for _p in ps
        @test all(isfinite.(gs[_p]))
    end

    # Testing SkipDEQ with no extra parameters
    Random.seed!(0)

    model = gpu(DEQChain(Dense(2, 2),
                         SkipDeepEquilibriumNetwork(Parallel(+, Dense(2, 2), Dense(2, 2)),
                                                    get_default_dynamicss_solver(0.1f0, 0.1f0);
                                                    sensealg=get_default_ssadjoint(0.1f0, 0.1f0, 10))))
    x = gpu(rand(Float32, 2, 1))
    y = gpu(rand(Float32, 2, 1))
    ps = Flux.params(model)
    gs = Flux.gradient(() -> mse_loss_function(model, x, y), ps)
    for _p in ps
        @test all(isfinite.(gs[_p]))
    end

    # Testing L-Broyden Solver
    Random.seed!(0)

    model = gpu(DEQChain(Conv((3, 3), 1 => 1, relu; pad=1, stride=1),
                         SkipDeepEquilibriumNetwork(Parallel(+, Conv((3, 3), 1 => 1, relu; pad=1, stride=1),
                                                             Conv((3, 3), 1 => 1, relu; pad=1, stride=1)),
                                                    Conv((3, 3), 1 => 1, relu; pad=1, stride=1),
                                                    get_default_ssrootfind_solver(0.1f0, 0.1f0,
                                                                                  LimitedMemoryBroydenSolver;
                                                                                  device=gpu, original_dims=(8 * 8, 1),
                                                                                  batch_size=4, abstol=1f-3);
                                                    sensealg=get_default_ssadjoint(0.1f0, 0.1f0, 10))))
    x = gpu(rand(Float32, 8, 8, 1, 4))
    y = gpu(rand(Float32, 8, 8, 1, 4))
    ps = Flux.params(model)
    gs = Flux.gradient(() -> mse_loss_function(model, x, y), ps)
    for _p in ps
        @test all(isfinite.(gs[_p]))
    end

    # Testing MultiScaleDEQ
    Random.seed!(0)

    model = gpu(MultiScaleDeepEquilibriumNetwork((Parallel(+, Dense(4, 4, tanh_fast), Dense(4, 4, tanh_fast)),
                                                  Dense(3, 3, tanh_fast), Dense(2, 2, tanh_fast),
                                                  Dense(1, 1, tanh_fast)),
                                                 [identity Dense(4, 3, tanh_fast) Dense(4, 2, tanh_fast) Dense(4, 1, tanh_fast);
                                                  Dense(3, 4, tanh_fast) identity Dense(3, 2, tanh_fast) Dense(3, 1, tanh_fast);
                                                  Dense(2, 4, tanh_fast) Dense(2, 3, tanh_fast) identity Dense(2, 1, tanh_fast);
                                                  Dense(1, 4, tanh_fast) Dense(1, 3, tanh_fast) Dense(1, 2, tanh_fast) identity],
                                                 get_default_dynamicss_solver(0.1f0, 0.1f0);
                                                 sensealg=get_default_ssadjoint(0.1f0, 0.1f0, 10), maxiters=10))
    x = gpu(rand(4, 2))
    y = tuple([gpu(rand(i, 2)) for i in 4:-1:1]...)
    sol = model(x)
    ps = Flux.params(model)
    gs = Flux.gradient(() -> mse_loss_function(model, x, y), ps)
    for _p in ps
        @test all(isfinite.(gs[_p]))
    end

    # Testing MultiScaleSkipDEQ
    Random.seed!(0)

    model = gpu(MultiScaleSkipDeepEquilibriumNetwork((Parallel(+, Dense(4, 4, tanh_fast), Dense(4, 4, tanh_fast)),
                                                      Dense(3, 3, tanh_fast), Dense(2, 2, tanh_fast),
                                                      Dense(1, 1, tanh_fast)),
                                                     [identity Dense(4, 3, tanh_fast) Dense(4, 2, tanh_fast) Dense(4, 1, tanh_fast);
                                                      Dense(3, 4, tanh_fast) identity Dense(3, 2, tanh_fast) Dense(3, 1, tanh_fast);
                                                      Dense(2, 4, tanh_fast) Dense(2, 3, tanh_fast) identity Dense(2, 1, tanh_fast);
                                                      Dense(1, 4, tanh_fast) Dense(1, 3, tanh_fast) Dense(1, 2, tanh_fast) identity],
                                                     (Dense(4, 4, tanh_fast), Dense(4, 3, tanh_fast),
                                                      Dense(4, 2, tanh_fast), Dense(4, 1, tanh_fast)),
                                                     get_default_ssrootfind_solver(0.1f0, 0.1f0,
                                                                                   LimitedMemoryBroydenSolver;
                                                                                   device=gpu, original_dims=(1, 10),
                                                                                   batch_size=2, maxiters=10);
                                                     sensealg=get_default_ssadjoint(0.1f0, 0.1f0, 10), maxiters=10))
    x = gpu(rand(4, 2))
    y = tuple([gpu(rand(i, 2)) for i in 4:-1:1]...)
    sol = model(x)
    ps = Flux.params(model)
    gs = Flux.gradient(() -> mse_loss_function(model, x, y), ps)
    for _p in ps
        @test all(isfinite.(gs[_p]))
    end

    # Testing MultiScaleSkipDEQ with no extra parameters
    Random.seed!(0)

    model = gpu(MultiScaleSkipDeepEquilibriumNetwork((Parallel(+, Dense(4, 4, tanh_fast), Dense(4, 4, tanh_fast)),
                                                      Dense(3, 3, tanh_fast), Dense(2, 2, tanh_fast),
                                                      Dense(1, 1, tanh_fast)),
                                                     [identity Dense(4, 3, tanh_fast) Dense(4, 2, tanh_fast) Dense(4, 1, tanh_fast);
                                                      Dense(3, 4, tanh_fast) identity Dense(3, 2, tanh_fast) Dense(3, 1, tanh_fast);
                                                      Dense(2, 4, tanh_fast) Dense(2, 3, tanh_fast) identity Dense(2, 1, tanh_fast);
                                                      Dense(1, 4, tanh_fast) Dense(1, 3, tanh_fast) Dense(1, 2, tanh_fast) identity],
                                                     get_default_ssrootfind_solver(0.1f0, 0.1f0,
                                                                                   LimitedMemoryBroydenSolver;
                                                                                   device=gpu, original_dims=(1, 10),
                                                                                   batch_size=2, maxiters=10);
                                                     sensealg=get_default_ssadjoint(0.1f0, 0.1f0, 10), maxiters=10))
    x = gpu(rand(4, 2))
    y = tuple([gpu(rand(i, 2)) for i in 4:-1:1]...)
    sol = model(x)
    ps = Flux.params(model)
    gs = Flux.gradient(() -> mse_loss_function(model, x, y), ps)
    for _p in ps
        @test all(isfinite.(gs[_p]))
    end

    # Testing WeightNorm Layer
    Random.seed!(0)

    model = gpu(DeepEquilibriumNetwork(Parallel(+, WeightNorm(Dense(2, 2; bias=false)),
                                                WeightNorm(Dense(2, 2; bias=false))),
                                       get_default_dynamicss_solver(0.1f0, 0.1f0);
                                       sensealg=get_default_ssadjoint(0.1f0, 0.1f0, 10)))
    x = gpu(rand(Float32, 2, 1))
    y = gpu(rand(Float32, 2, 1))
    ps = Flux.params(model)
    gs = Flux.gradient(() -> mse_loss_function(model, x, y), ps)
    for _p in ps
        @test all(isfinite.(gs[_p]))
    end
end
