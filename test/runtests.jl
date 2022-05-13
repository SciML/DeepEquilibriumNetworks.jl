using CUDA, FastDEQ, Functors, LinearAlgebra, Lux, Random, Test, Zygote

function test_gradient_isfinite(gs::NamedTuple)
    gradient_is_finite = [true]
    function is_gradient_finite(x)
        if !isnothing(x) && !all(isfinite, x)
            gradient_is_finite[1] = false
        end
        return x
    end
    fmap(is_gradient_finite, gs)
    return gradient_is_finite[1]
end

@testset "FastDEQ.jl" begin
    seed = 0
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    @info "Testing DEQ"
    model = DEQChain(
        Dense(2, 2),
        DeepEquilibriumNetwork(
            Parallel(+, Dense(2, 2; bias=false), Dense(2, 2; bias=false)),
            ContinuousDEQSolver(; abstol=0.1f0, reltol=0.1f0, abstol_termination=0.1f0, reltol_termination=0.1f0),
        ),
    )
    ps, st = gpu.(Lux.setup(rng, model))
    x = gpu(rand(rng, Float32, 2, 1))
    y = gpu(rand(rng, Float32, 2, 1))

    @inferred model(x, ps, st)

    gs = gradient(p -> sum(abs2, model(x, p, st)[1][1] .- y), ps)[1]

    @test test_gradient_isfinite(gs)

    @info "Testing DEQ without Fixed Point Iterations"
    st = Lux.update_state(st, :fixed_depth, Val(5))

    @inferred model(x, ps, st)

    gs = gradient(p -> sum(abs2, model(x, p, st)[1][1] .- y), ps)[1]

    @test test_gradient_isfinite(gs)

    @info "Testing SkipDEQ"
    Random.seed!(rng, seed)
    model = DEQChain(
        Dense(2, 2),
        SkipDeepEquilibriumNetwork(
            Parallel(+, Dense(2, 2), Dense(2, 2)),
            Dense(2, 2),
            ContinuousDEQSolver(; abstol=0.1f0, reltol=0.1f0, abstol_termination=0.1f0, reltol_termination=0.1f0);
            sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10),
        ),
    )
    ps, st = gpu.(Lux.setup(rng, model))
    x = gpu(rand(rng, Float32, 2, 1))
    y = gpu(rand(rng, Float32, 2, 1))
    
    @inferred model(x, ps, st)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(abs2, ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end[1]

    @test test_gradient_isfinite(gs)

    @info "Testing SkipDEQ without Fixed Point Iterations"
    st = Lux.update_state(st, :fixed_depth, Val(5))
    
    @inferred model(x, ps, st)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(abs2, ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end[1]

    @test test_gradient_isfinite(gs)

    @info "Testing SkipDEQV2"
    Random.seed!(rng, seed)
    model = DEQChain(
        Dense(2, 2),
        SkipDeepEquilibriumNetwork(
            Parallel(+, Dense(2, 2), Dense(2, 2)),
            nothing,
            ContinuousDEQSolver(; abstol=0.1f0, reltol=0.1f0, abstol_termination=0.1f0, reltol_termination=0.1f0);
            sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10),
        ),
    )
    ps, st = gpu.(Lux.setup(rng, model))
    x = gpu(rand(rng, Float32, 2, 1))
    y = gpu(rand(rng, Float32, 2, 1))
    
    @inferred model(x, ps, st)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(abs2, ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end[1]

    @test test_gradient_isfinite(gs)

    @info "Testing SkipDEQV2 without Fixed Point Iterations"
    st = Lux.update_state(st, :fixed_depth, Val(5))
    
    @inferred model(x, ps, st)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(abs2, ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end[1]

    @test test_gradient_isfinite(gs)

    @info "Testing SkipDEQ with Broyden Solver"
    Random.seed!(rng, seed)
    model = DEQChain(
        Dense(2, 2),
        SkipDeepEquilibriumNetwork(
            Parallel(+, Dense(2, 2), Dense(2, 2)),
            Dense(2, 2),
            DiscreteDEQSolver(BroydenSolver(); abstol_termination=0.1f0, reltol_termination=0.1f0);
            sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10),
        ),
    )
    ps, st = gpu.(Lux.setup(rng, model))
    x = gpu(rand(rng, Float32, 2, 1))
    y = gpu(rand(rng, Float32, 2, 1))
    
    @inferred model(x, ps, st)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(abs2, ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end[1]

    @test test_gradient_isfinite(gs)

    @info "Testing SkipDEQ with L-Broyden Solver"
    Random.seed!(rng, seed)
    model = DEQChain(
        Dense(2, 2),
        SkipDeepEquilibriumNetwork(
            Parallel(+, Dense(2, 2), Dense(2, 2)),
            Dense(2, 2),
            DiscreteDEQSolver(LimitedMemoryBroydenSolver(); abstol_termination=0.1f0, reltol_termination=0.1f0);
            sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10),
        ),
    )
    ps, st = gpu.(Lux.setup(rng, model))
    x = gpu(rand(rng, Float32, 2, 1))
    y = gpu(rand(rng, Float32, 2, 1))
    
    @inferred model(x, ps, st)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(abs2, ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end[1]

    @test test_gradient_isfinite(gs)

    @info "Testing MultiScaleDEQ"
    Random.seed!(rng, seed)
    model = MultiScaleDeepEquilibriumNetwork(
        (
            Parallel(+, Dense(4, 4, tanh), Dense(4, 4, tanh)),
            Dense(3, 3, tanh),
            Dense(2, 2, tanh),
            Dense(1, 1, tanh),
        ),
        [
            NoOpLayer() Dense(4, 3, tanh) Dense(4, 2, tanh) Dense(4, 1, tanh)
            Dense(3, 4, tanh) NoOpLayer() Dense(3, 2, tanh) Dense(3, 1, tanh)
            Dense(2, 4, tanh) Dense(2, 3, tanh) NoOpLayer() Dense(2, 1, tanh)
            Dense(1, 4, tanh) Dense(1, 3, tanh) Dense(1, 2, tanh) NoOpLayer()
        ],
        nothing,
        ContinuousDEQSolver(; abstol=0.1f0, reltol=0.1f0, abstol_termination=0.1f0, reltol_termination=0.1f0),
        ((4,), (3,), (2,), (1,));
        sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10),
    )

    ps, st = gpu.(Lux.setup(rng, model))
    x = gpu(rand(rng, Float32, 4, 2))
    y = tuple([gpu(rand(rng, Float32, i, 2)) for i in 4:-1:1]...)
    
    @inferred model(x, ps, st)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(Base.Fix1(sum, abs2), ŷ .- y)
    end[1]

    @test test_gradient_isfinite(gs)

    @info "Testing MultiScaleDEQ without Fixed Point Iterations"
    st = Lux.update_state(st, :fixed_depth, Val(5))
    
    @inferred model(x, ps, st)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(Base.Fix1(sum, abs2), ŷ .- y)
    end[1]

    @test test_gradient_isfinite(gs)

    @info "Testing MultiScaleSkipDEQ"
    Random.seed!(rng, seed)
    model = MultiScaleSkipDeepEquilibriumNetwork(
        (
            Parallel(+, Dense(4, 4, tanh), Dense(4, 4, tanh)),
            Dense(3, 3, tanh),
            Dense(2, 2, tanh),
            Dense(1, 1, tanh),
        ),
        [
            NoOpLayer() Dense(4, 3, tanh) Dense(4, 2, tanh) Dense(4, 1, tanh)
            Dense(3, 4, tanh) NoOpLayer() Dense(3, 2, tanh) Dense(3, 1, tanh)
            Dense(2, 4, tanh) Dense(2, 3, tanh) NoOpLayer() Dense(2, 1, tanh)
            Dense(1, 4, tanh) Dense(1, 3, tanh) Dense(1, 2, tanh) NoOpLayer()
        ],
        nothing,
        (Dense(4, 4, tanh), Dense(4, 3, tanh), Dense(4, 2, tanh), Dense(4, 1, tanh)),
        ContinuousDEQSolver(; abstol=0.1f0, reltol=0.1f0, abstol_termination=0.1f0, reltol_termination=0.1f0),
        ((4,), (3,), (2,), (1,));
        sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10),
    )

    ps, st = gpu.(Lux.setup(rng, model))
    x = gpu(rand(rng, Float32, 4, 2))
    y = tuple([gpu(rand(rng, Float32, i, 2)) for i in 4:-1:1]...)
    
    @inferred model(x, ps, st)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(Base.Fix1(sum, abs2), ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end[1]

    @test test_gradient_isfinite(gs)

    @info "Testing MultiScaleSkipDEQ without Fixed Point Iterations"
    st = Lux.update_state(st, :fixed_depth, Val(5))
    
    @inferred model(x, ps, st)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(Base.Fix1(sum, abs2), ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end[1]

    @test test_gradient_isfinite(gs)

    @info "Testing MultiScaleSkipDEQV2"
    Random.seed!(rng, seed)
    model = MultiScaleSkipDeepEquilibriumNetwork(
        (
            Parallel(+, Dense(4, 4, tanh), Dense(4, 4, tanh)),
            Dense(3, 3, tanh),
            Dense(2, 2, tanh),
            Dense(1, 1, tanh),
        ),
        [
            NoOpLayer() Dense(4, 3, tanh) Dense(4, 2, tanh) Dense(4, 1, tanh)
            Dense(3, 4, tanh) NoOpLayer() Dense(3, 2, tanh) Dense(3, 1, tanh)
            Dense(2, 4, tanh) Dense(2, 3, tanh) NoOpLayer() Dense(2, 1, tanh)
            Dense(1, 4, tanh) Dense(1, 3, tanh) Dense(1, 2, tanh) NoOpLayer()
        ],
        nothing,
        nothing,
        ContinuousDEQSolver(; abstol=0.1f0, reltol=0.1f0, abstol_termination=0.1f0, reltol_termination=0.1f0),
        ((4,), (3,), (2,), (1,));
        sensealg=DeepEquilibriumAdjoint(0.1f0, 0.1f0, 10),
    )

    ps, st = gpu.(Lux.setup(rng, model))
    x = gpu(rand(rng, Float32, 4, 2))
    y = tuple([gpu(rand(rng, Float32, i, 2)) for i in 4:-1:1]...)
    
    @inferred model(x, ps, st)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(Base.Fix1(sum, abs2), ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end[1]

    @test test_gradient_isfinite(gs)

    @info "Testing MultiScaleSkipDEQV2 without Fixed Point Iterations"
    st = Lux.update_state(st, :fixed_depth, Val(5))
    
    @inferred model(x, ps, st)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(Base.Fix1(sum, abs2), ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end[1]

    @test test_gradient_isfinite(gs)
end
