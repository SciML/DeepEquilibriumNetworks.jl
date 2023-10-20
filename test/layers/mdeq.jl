using DeepEquilibriumNetworks
using Test

include("../test_utils.jl")

function DEFAULT_MDEQ_SOLVERS()
    termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.RelSafe;
        abstol=0.01f0, reltol=0.01f0)

    return (ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0),
        DiscreteDEQSolver(LBroyden(; batched=true, termination_condition)))
end

function test_multiscale_deep_equilibrium_network()
    rng = get_prng(0)

    main_layers = (Parallel(+, get_dense_layer(4, 4), get_dense_layer(4, 4)),
        get_dense_layer(3, 3),
        get_dense_layer(2, 2),
        get_dense_layer(1, 1))

    mapping_layers = [NoOpLayer() get_dense_layer(4, 3) get_dense_layer(4, 2) get_dense_layer(4, 1);
        get_dense_layer(3, 4) NoOpLayer() get_dense_layer(3, 2) get_dense_layer(3, 1);
        get_dense_layer(2, 4) get_dense_layer(2, 3) NoOpLayer() get_dense_layer(2, 1);
        get_dense_layer(1, 4) get_dense_layer(1, 3) get_dense_layer(1, 2) NoOpLayer()]

    for solver in DEFAULT_MDEQ_SOLVERS()
        scales = ((4,), (3,), (2,), (1,))
        model = MultiScaleDeepEquilibriumNetwork(main_layers, mapping_layers, nothing,
            solver, scales; save_everystep=true)

        ps, st = Lux.setup(rng, model)

        @test st.solution === nothing

        x = randn(rng, Float32, 4, 1)

        z, st = model(x, ps, st)

        @test all(Base.Fix1(all, isfinite), z)
        @test all(map(x -> size(x)[1:(end - 1)], z) .== scales)
        @test st.solution isa DeepEquilibriumSolution

        ps, st = Lux.setup(rng, model)
        st = Lux.update_state(st, :fixed_depth, Val(10))

        @test st.solution === nothing

        z, st = model(x, ps, st)

        @test all(Base.Fix1(all, isfinite), z)
        @test all(map(x -> size(x)[1:(end - 1)], z) .== scales)
        @test st.solution isa DeepEquilibriumSolution
        @test st.solution.nfe == 10
    end

    return nothing
end

function test_multiscale_skip_deep_equilibrium_network()
    rng = get_prng(0)

    shortcut_layers = (get_dense_layer(4, 4), get_dense_layer(4, 3), get_dense_layer(4, 2),
        get_dense_layer(4, 1))

    main_layers = (Parallel(+, get_dense_layer(4, 4), get_dense_layer(4, 4)),
        get_dense_layer(3, 3), get_dense_layer(2, 2), get_dense_layer(1, 1))

    mapping_layers = [NoOpLayer() get_dense_layer(4, 3) get_dense_layer(4, 2) get_dense_layer(4, 1);
        get_dense_layer(3, 4) NoOpLayer() get_dense_layer(3, 2) get_dense_layer(3, 1);
        get_dense_layer(2, 4) get_dense_layer(2, 3) NoOpLayer() get_dense_layer(2, 1);
        get_dense_layer(1, 4) get_dense_layer(1, 3) get_dense_layer(1, 2) NoOpLayer()]

    for solver in DEFAULT_MDEQ_SOLVERS()
        scales = ((4,), (3,), (2,), (1,))
        model = MultiScaleSkipDeepEquilibriumNetwork(main_layers, mapping_layers, nothing,
            shortcut_layers, solver, scales; save_everystep=true)

        ps, st = Lux.setup(rng, model)

        @test st.solution === nothing

        x = randn(rng, Float32, 4, 1)

        z, st = model(x, ps, st)

        @test all(Base.Fix1(all, isfinite), z)
        @test all(map(x -> size(x)[1:(end - 1)], z) .== scales)
        @test st.solution isa DeepEquilibriumSolution

        ps, st = Lux.setup(rng, model)
        st = Lux.update_state(st, :fixed_depth, Val(10))

        @test st.solution === nothing

        z, st = model(x, ps, st)

        @test all(Base.Fix1(all, isfinite), z)
        @test all(map(x -> size(x)[1:(end - 1)], z) .== scales)
        @test st.solution isa DeepEquilibriumSolution
        @test st.solution.nfe == 10
    end

    return nothing
end

function test_multiscale_skip_deep_equilibrium_network_v2()
    rng = get_prng(0)

    main_layers = (Parallel(+, get_dense_layer(4, 4), get_dense_layer(4, 4)),
        get_dense_layer(3, 3),
        get_dense_layer(2, 2),
        get_dense_layer(1, 1))

    mapping_layers = [NoOpLayer() get_dense_layer(4, 3) get_dense_layer(4, 2) get_dense_layer(4, 1);
        get_dense_layer(3, 4) NoOpLayer() get_dense_layer(3, 2) get_dense_layer(3, 1);
        get_dense_layer(2, 4) get_dense_layer(2, 3) NoOpLayer() get_dense_layer(2, 1);
        get_dense_layer(1, 4) get_dense_layer(1, 3) get_dense_layer(1, 2) NoOpLayer()]

    for solver in DEFAULT_MDEQ_SOLVERS()
        scales = ((4,), (3,), (2,), (1,))
        model = DEQs.MultiScaleSkipDeepEquilibriumNetwork(main_layers, mapping_layers,
            nothing, nothing, solver, scales; save_everystep=true)

        ps, st = Lux.setup(rng, model)

        @test st.solution === nothing

        x = randn(rng, Float32, 4, 1)

        z, st = model(x, ps, st)

        @test all(Base.Fix1(all, isfinite), z)
        @test all(map(x -> size(x)[1:(end - 1)], z) .== scales)
        @test st.solution isa DeepEquilibriumSolution

        ps, st = Lux.setup(rng, model)
        st = Lux.update_state(st, :fixed_depth, Val(10))

        @test st.solution === nothing

        z, st = model(x, ps, st)

        @test all(Base.Fix1(all, isfinite), z)
        @test all(map(x -> size(x)[1:(end - 1)], z) .== scales)
        @test st.solution isa DeepEquilibriumSolution
        @test st.solution.nfe == 10
    end

    return nothing
end

function test_multiscale_neural_ode()
    rng = get_prng(0)

    solver = VCABM3()

    main_layers = (Parallel(+, get_dense_layer(4, 4), get_dense_layer(4, 4)),
        get_dense_layer(3, 3),
        get_dense_layer(2, 2),
        get_dense_layer(1, 1))

    mapping_layers = [NoOpLayer() get_dense_layer(4, 3) get_dense_layer(4, 2) get_dense_layer(4, 1);
        get_dense_layer(3, 4) NoOpLayer() get_dense_layer(3, 2) get_dense_layer(3, 1);
        get_dense_layer(2, 4) get_dense_layer(2, 3) NoOpLayer() get_dense_layer(2, 1);
        get_dense_layer(1, 4) get_dense_layer(1, 3) get_dense_layer(1, 2) NoOpLayer()]

    scales = ((4,), (3,), (2,), (1,))
    model = MultiScaleNeuralODE(main_layers, mapping_layers, nothing, solver, scales;
        abstol=0.01f0, reltol=0.01f0)

    ps, st = Lux.setup(rng, model)

    @test st.solution === nothing

    x = randn(rng, Float32, 4, 1)

    z, st = model(x, ps, st)

    @test all(Base.Fix1(all, isfinite), z)
    @test all(map(x -> size(x)[1:(end - 1)], z) .== scales)
    @test st.solution isa DeepEquilibriumSolution

    ps, st = Lux.setup(rng, model)
    st = Lux.update_state(st, :fixed_depth, Val(10))

    @test st.solution === nothing

    z, st = model(x, ps, st)

    @test all(Base.Fix1(all, isfinite), z)
    @test all(map(x -> size(x)[1:(end - 1)], z) .== scales)
    @test st.solution isa DeepEquilibriumSolution
    @test st.solution.nfe == 10

    return nothing
end

@testset "MultiScaleDeepEquilibriumNetwork" begin
    test_multiscale_deep_equilibrium_network()
end
@testset "MultiScaleSkipDeepEquilibriumNetwork" begin
    test_multiscale_skip_deep_equilibrium_network()
end
@testset "MultiScaleSkipRegDeepEquilibriumNetwork" begin
    test_multiscale_skip_deep_equilibrium_network_v2()
end
@testset "MultiScaleNeuralODE" begin
    test_multiscale_neural_ode()
end
