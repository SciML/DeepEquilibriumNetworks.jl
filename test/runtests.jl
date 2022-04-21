using FastDEQ, CUDA, LinearAlgebra, Random, Test, ExplicitFluxLayers, Flux

@testset "FastDEQ.jl" begin
    seed = 0

    @info "Testing DEQ"
    model = DEQChain(
        EFL.Dense(2, 2),
        DeepEquilibriumNetwork(
            EFL.Parallel(+, EFL.Dense(2, 2; bias=false), EFL.Dense(2, 2; bias=false)),
            ContinuousDEQSolver(; abstol=0.1f0, reltol=0.1f0, abstol_termination=0.1f0, reltol_termination=0.1f0),
        ),
    )
    ps, st = NamedTuple.(gpu.(EFL.setup(MersenneTwister(seed), model)))
    x = gpu(rand(MersenneTwister(seed + 1), Float32, 2, 1))
    y = gpu(rand(MersenneTwister(seed + 2), Float32, 2, 1))

    gs = gradient(p -> sum(abs2, model(x, p, st)[1][1] .- y), ps)

    @info "Testing DEQ without Fixed Point Iterations"
    st = EFL.update_state(st, :fixed_depth, 5)

    gs = gradient(p -> sum(abs2, model(x, p, st)[1][1] .- y), ps)

    @info "Testing SkipDEQ"
    model = DEQChain(
        EFL.Dense(2, 2),
        SkipDeepEquilibriumNetwork(
            EFL.Parallel(+, EFL.Dense(2, 2), EFL.Dense(2, 2)),
            EFL.Dense(2, 2),
            ContinuousDEQSolver(; abstol=0.1f0, reltol=0.1f0, abstol_termination=0.1f0, reltol_termination=0.1f0);
            sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10),
        ),
    )
    ps, st = NamedTuple.(gpu.(EFL.setup(MersenneTwister(seed), model)))
    x = gpu(rand(MersenneTwister(seed + 1), Float32, 2, 1))
    y = gpu(rand(MersenneTwister(seed + 2), Float32, 2, 1))

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(abs2, ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end

    @info "Testing SkipDEQ without Fixed Point Iterations"
    st = EFL.update_state(st, :fixed_depth, 5)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(abs2, ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end

    @info "Testing SkipDEQV2"
    model = DEQChain(
        EFL.Dense(2, 2),
        SkipDeepEquilibriumNetwork(
            EFL.Parallel(+, EFL.Dense(2, 2), EFL.Dense(2, 2)),
            nothing,
            ContinuousDEQSolver(; abstol=0.1f0, reltol=0.1f0, abstol_termination=0.1f0, reltol_termination=0.1f0);
            sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10),
        ),
    )
    ps, st = NamedTuple.(gpu.(EFL.setup(MersenneTwister(seed), model)))
    x = gpu(rand(MersenneTwister(seed + 1), Float32, 2, 1))
    y = gpu(rand(MersenneTwister(seed + 2), Float32, 2, 1))

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(abs2, ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end

    @info "Testing SkipDEQV2 without Fixed Point Iterations"
    st = EFL.update_state(st, :fixed_depth, 5)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(abs2, ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end

    @info "Testing SkipDEQ with Broyden Solver"
    model = DEQChain(
        EFL.Dense(2, 2),
        SkipDeepEquilibriumNetwork(
            EFL.Parallel(+, EFL.Dense(2, 2), EFL.Dense(2, 2)),
            EFL.Dense(2, 2),
            DiscreteDEQSolver(BroydenSolver(); abstol_termination=0.1f0, reltol_termination=0.1f0);
            sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10),
        ),
    )
    ps, st = NamedTuple.(gpu.(EFL.setup(MersenneTwister(seed), model)))
    x = gpu(rand(MersenneTwister(seed + 1), Float32, 2, 1))
    y = gpu(rand(MersenneTwister(seed + 2), Float32, 2, 1))

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(abs2, ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end

    @info "Testing SkipDEQ with L-Broyden Solver"
    model = DEQChain(
        EFL.Dense(2, 2),
        SkipDeepEquilibriumNetwork(
            EFL.Parallel(+, EFL.Dense(2, 2), EFL.Dense(2, 2)),
            EFL.Dense(2, 2),
            DiscreteDEQSolver(LimitedMemoryBroydenSolver(); abstol_termination=0.1f0, reltol_termination=0.1f0);
            sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10),
        ),
    )
    ps, st = NamedTuple.(gpu.(EFL.setup(MersenneTwister(seed), model)))
    x = gpu(rand(MersenneTwister(seed + 1), Float32, 2, 1))
    y = gpu(rand(MersenneTwister(seed + 2), Float32, 2, 1))

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(abs2, ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end

    @info "Testing MultiScaleDEQ"
    model = MultiScaleDeepEquilibriumNetwork(
        (
            EFL.Parallel(+, EFL.Dense(4, 4, tanh), EFL.Dense(4, 4, tanh)),
            EFL.Dense(3, 3, tanh),
            EFL.Dense(2, 2, tanh),
            EFL.Dense(1, 1, tanh),
        ),
        [
            EFL.NoOpLayer() EFL.Dense(4, 3, tanh) EFL.Dense(4, 2, tanh) EFL.Dense(4, 1, tanh)
            EFL.Dense(3, 4, tanh) EFL.NoOpLayer() EFL.Dense(3, 2, tanh) EFL.Dense(3, 1, tanh)
            EFL.Dense(2, 4, tanh) EFL.Dense(2, 3, tanh) EFL.NoOpLayer() EFL.Dense(2, 1, tanh)
            EFL.Dense(1, 4, tanh) EFL.Dense(1, 3, tanh) EFL.Dense(1, 2, tanh) EFL.NoOpLayer()
        ],
        nothing,
        ContinuousDEQSolver(; abstol=0.1f0, reltol=0.1f0, abstol_termination=0.1f0, reltol_termination=0.1f0),
        ((4,), (3,), (2,), (1,));
        sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10),
    )

    ps, st = NamedTuple.(gpu.(EFL.setup(MersenneTwister(seed), model)))
    x = gpu(rand(MersenneTwister(seed + 1), Float32, 4, 2))
    y = tuple([gpu(rand(MersenneTwister(seed + 1 + i), Float32, i, 2)) for i in 4:-1:1]...)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(Base.Fix1(sum, abs2), ŷ .- y)
    end

    @info "Testing MultiScaleDEQ without Fixed Point Iterations"
    st = EFL.update_state(st, :fixed_depth, 5)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(Base.Fix1(sum, abs2), ŷ .- y)
    end

    @info "Testing MultiScaleSkipDEQ"
    model = MultiScaleSkipDeepEquilibriumNetwork(
        (
            EFL.Parallel(+, EFL.Dense(4, 4, tanh), EFL.Dense(4, 4, tanh)),
            EFL.Dense(3, 3, tanh),
            EFL.Dense(2, 2, tanh),
            EFL.Dense(1, 1, tanh),
        ),
        [
            EFL.NoOpLayer() EFL.Dense(4, 3, tanh) EFL.Dense(4, 2, tanh) EFL.Dense(4, 1, tanh)
            EFL.Dense(3, 4, tanh) EFL.NoOpLayer() EFL.Dense(3, 2, tanh) EFL.Dense(3, 1, tanh)
            EFL.Dense(2, 4, tanh) EFL.Dense(2, 3, tanh) EFL.NoOpLayer() EFL.Dense(2, 1, tanh)
            EFL.Dense(1, 4, tanh) EFL.Dense(1, 3, tanh) EFL.Dense(1, 2, tanh) EFL.NoOpLayer()
        ],
        nothing,
        (EFL.Dense(4, 4, tanh), EFL.Dense(4, 3, tanh), EFL.Dense(4, 2, tanh), EFL.Dense(4, 1, tanh)),
        ContinuousDEQSolver(; abstol=0.1f0, reltol=0.1f0, abstol_termination=0.1f0, reltol_termination=0.1f0),
        ((4,), (3,), (2,), (1,));
        sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10),
    )

    ps, st = NamedTuple.(gpu.(EFL.setup(MersenneTwister(seed), model)))
    x = gpu(rand(MersenneTwister(seed + 1), Float32, 4, 2))
    y = tuple([gpu(rand(MersenneTwister(seed + 1 + i), Float32, i, 2)) for i in 4:-1:1]...)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(Base.Fix1(sum, abs2), ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end

    @info "Testing MultiScaleSkipDEQ without Fixed Point Iterations"
    st = EFL.update_state(st, :fixed_depth, 5)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(Base.Fix1(sum, abs2), ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end

    @info "Testing MultiScaleSkipDEQV2"
    model = MultiScaleSkipDeepEquilibriumNetwork(
        (
            EFL.Parallel(+, EFL.Dense(4, 4, tanh), EFL.Dense(4, 4, tanh)),
            EFL.Dense(3, 3, tanh),
            EFL.Dense(2, 2, tanh),
            EFL.Dense(1, 1, tanh),
        ),
        [
            EFL.NoOpLayer() EFL.Dense(4, 3, tanh) EFL.Dense(4, 2, tanh) EFL.Dense(4, 1, tanh)
            EFL.Dense(3, 4, tanh) EFL.NoOpLayer() EFL.Dense(3, 2, tanh) EFL.Dense(3, 1, tanh)
            EFL.Dense(2, 4, tanh) EFL.Dense(2, 3, tanh) EFL.NoOpLayer() EFL.Dense(2, 1, tanh)
            EFL.Dense(1, 4, tanh) EFL.Dense(1, 3, tanh) EFL.Dense(1, 2, tanh) EFL.NoOpLayer()
        ],
        nothing,
        nothing,
        ContinuousDEQSolver(; abstol=0.1f0, reltol=0.1f0, abstol_termination=0.1f0, reltol_termination=0.1f0),
        ((4,), (3,), (2,), (1,));
        sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10),
    )

    ps, st = NamedTuple.(gpu.(EFL.setup(MersenneTwister(seed), model)))
    x = gpu(rand(MersenneTwister(seed + 1), Float32, 4, 2))
    y = tuple([gpu(rand(MersenneTwister(seed + 1 + i), Float32, i, 2)) for i in 4:-1:1]...)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(Base.Fix1(sum, abs2), ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end

    @info "Testing MultiScaleSkipDEQV2 without Fixed Point Iterations"
    st = EFL.update_state(st, :fixed_depth, 5)

    gs = gradient(ps) do p
        (ŷ, soln), _ = model(x, p, st)
        sum(Base.Fix1(sum, abs2), ŷ .- y) + sum(abs2, soln.u₀ .- soln.z_star)
    end
end
