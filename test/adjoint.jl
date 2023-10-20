using DeepEquilibriumNetworks, Statistics, Zygote
using Test

include("test_utils.jl")

function loss_function(model::DEQs.AbstractDeepEquilibriumNetwork, x, ps, st)
    y, st_ = model(x, ps, st)
    return sum(y) + st_.solution.jacobian_loss
end

function loss_function(model::DEQs.AbstractSkipDeepEquilibriumNetwork, x, ps, st)
    y, st_ = model(x, ps, st)
    return sum(y) + st_.solution.jacobian_loss +
           mean(abs, st_.solution.z_star .- st_.solution.u0)
end

function loss_function(model::Union{MultiScaleDeepEquilibriumNetwork, MultiScaleNeuralODE},
    x, ps, st)
    y, st_ = model(x, ps, st)
    return sum(sum, y) + st_.solution.jacobian_loss
end

function loss_function(model::MultiScaleSkipDeepEquilibriumNetwork, x, ps, st)
    y, st_ = model(x, ps, st)
    return sum(sum, y) + st_.solution.jacobian_loss +
           mean(abs, st_.solution.z_star .- st_.solution.u0)
end

function DEFAULT_DEQ_SOLVERS()
    termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.RelSafe;
        abstol=0.01f0, reltol=0.01f0)

    return (ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0),
        DiscreteDEQSolver(LBroyden(; batched=true, termination_condition)))
end

function test_deep_equilibrium_network_adjoint()
    rng = get_prng(0)

    for solver in DEFAULT_DEQ_SOLVERS(), jacobian_regularization in (true, false)
        sensealg = SteadyStateAdjoint()
        model = DeepEquilibriumNetwork(Parallel(+,
                get_dense_layer(2, 2; use_bias=false),
                get_dense_layer(2, 2; use_bias=false)),
            solver; sensealg, jacobian_regularization, verbose=false, save_everystep=true)

        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, 2, 1)

        gs = Zygote.gradient(loss_function, model, x, ps, st)

        @test is_finite_gradient(gs[2])
        @test is_finite_gradient(gs[3])

        st = Lux.update_state(st, :fixed_depth, Val(10))

        gs = Zygote.gradient(loss_function, model, x, ps, st)

        @test is_finite_gradient(gs[2])
        @test is_finite_gradient(gs[3])
    end

    return nothing
end

function test_skip_deep_equilibrium_network_adjoint()
    rng = get_prng(0)

    for solver in DEFAULT_DEQ_SOLVERS(), jacobian_regularization in (true, false)
        sensealg = SteadyStateAdjoint()
        model = SkipDeepEquilibriumNetwork(Parallel(+,
                get_dense_layer(2, 2; use_bias=false),
                get_dense_layer(2, 2; use_bias=false)),
            get_dense_layer(2, 2),
            solver; sensealg, jacobian_regularization, verbose=false, save_everystep=true)

        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, 2, 1)

        gs = Zygote.gradient(loss_function, model, x, ps, st)

        @test is_finite_gradient(gs[2])
        @test is_finite_gradient(gs[3])

        st = Lux.update_state(st, :fixed_depth, Val(10))

        gs = Zygote.gradient(loss_function, model, x, ps, st)

        @test is_finite_gradient(gs[2])
        @test is_finite_gradient(gs[3])
    end

    return nothing
end

function test_skip_reg_deep_equilibrium_network_adjoint()
    rng = get_prng(0)

    for solver in DEFAULT_DEQ_SOLVERS(), jacobian_regularization in (true, false)
        sensealg = SteadyStateAdjoint()
        model = SkipDeepEquilibriumNetwork(Parallel(+,
                get_dense_layer(2, 2; use_bias=false),
                get_dense_layer(2, 2; use_bias=false)),
            nothing, solver; sensealg, jacobian_regularization, verbose=false,
            save_everystep=true)

        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, 2, 1)

        gs = Zygote.gradient(loss_function, model, x, ps, st)

        @test is_finite_gradient(gs[2])
        @test is_finite_gradient(gs[3])

        st = Lux.update_state(st, :fixed_depth, Val(10))

        gs = Zygote.gradient(loss_function, model, x, ps, st)

        @test is_finite_gradient(gs[2])
        @test is_finite_gradient(gs[3])
    end

    return nothing
end

function test_multiscale_deep_equilibrium_network_adjoint()
    rng = get_prng(0)

    main_layers = (Parallel(+, get_dense_layer(4, 4), get_dense_layer(4, 4)),
        get_dense_layer(3, 3),
        get_dense_layer(2, 2),
        get_dense_layer(1, 1))

    mapping_layers = [NoOpLayer() get_dense_layer(4, 3) get_dense_layer(4, 2) get_dense_layer(4, 1);
        get_dense_layer(3, 4) NoOpLayer() get_dense_layer(3, 2) get_dense_layer(3, 1);
        get_dense_layer(2, 4) get_dense_layer(2, 3) NoOpLayer() get_dense_layer(2, 1);
        get_dense_layer(1, 4) get_dense_layer(1, 3) get_dense_layer(1, 2) NoOpLayer()]

    for solver in DEFAULT_DEQ_SOLVERS()
        sensealg = SteadyStateAdjoint()
        scales = ((4,), (3,), (2,), (1,))
        model = MultiScaleDeepEquilibriumNetwork(main_layers, mapping_layers, nothing,
            solver, scales; sensealg, verbose=false, save_everystep=true)

        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, 4, 1)

        gs = Zygote.gradient(loss_function, model, x, ps, st)

        @test is_finite_gradient(gs[2])
        @test is_finite_gradient(gs[3])

        st = Lux.update_state(st, :fixed_depth, Val(10))

        gs = Zygote.gradient(loss_function, model, x, ps, st)

        @test is_finite_gradient(gs[2])
        @test is_finite_gradient(gs[3])
    end

    return nothing
end

function test_multiscale_skip_deep_equilibrium_network_adjoint()
    rng = get_prng(0)

    main_layers = (Parallel(+, get_dense_layer(4, 4), get_dense_layer(4, 4)),
        get_dense_layer(3, 3),
        get_dense_layer(2, 2),
        get_dense_layer(1, 1))

    mapping_layers = [NoOpLayer() get_dense_layer(4, 3) get_dense_layer(4, 2) get_dense_layer(4, 1);
        get_dense_layer(3, 4) NoOpLayer() get_dense_layer(3, 2) get_dense_layer(3, 1);
        get_dense_layer(2, 4) get_dense_layer(2, 3) NoOpLayer() get_dense_layer(2, 1);
        get_dense_layer(1, 4) get_dense_layer(1, 3) get_dense_layer(1, 2) NoOpLayer()]

    shortcut_layers = (get_dense_layer(4, 4),
        get_dense_layer(4, 3),
        get_dense_layer(4, 2),
        get_dense_layer(4, 1))

    for solver in DEFAULT_DEQ_SOLVERS()
        sensealg = SteadyStateAdjoint()
        scales = ((4,), (3,), (2,), (1,))
        model = MultiScaleSkipDeepEquilibriumNetwork(main_layers, mapping_layers, nothing,
            shortcut_layers, solver, scales; sensealg, verbose=false, save_everystep=true)

        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, 4, 1)

        gs = Zygote.gradient(loss_function, model, x, ps, st)

        @test is_finite_gradient(gs[2])
        @test is_finite_gradient(gs[3])

        st = Lux.update_state(st, :fixed_depth, Val(10))

        gs = Zygote.gradient(loss_function, model, x, ps, st)

        @test is_finite_gradient(gs[2])
        @test is_finite_gradient(gs[3])
    end

    return nothing
end

function test_multiscale_skip_reg_deep_equilibrium_network_adjoint()
    rng = get_prng(0)

    main_layers = (Parallel(+, get_dense_layer(4, 4), get_dense_layer(4, 4)),
        get_dense_layer(3, 3),
        get_dense_layer(2, 2),
        get_dense_layer(1, 1))

    mapping_layers = [NoOpLayer() get_dense_layer(4, 3) get_dense_layer(4, 2) get_dense_layer(4, 1);
        get_dense_layer(3, 4) NoOpLayer() get_dense_layer(3, 2) get_dense_layer(3, 1);
        get_dense_layer(2, 4) get_dense_layer(2, 3) NoOpLayer() get_dense_layer(2, 1);
        get_dense_layer(1, 4) get_dense_layer(1, 3) get_dense_layer(1, 2) NoOpLayer()]

    for solver in DEFAULT_DEQ_SOLVERS()
        sensealg = SteadyStateAdjoint()
        scales = ((4,), (3,), (2,), (1,))
        model = MultiScaleSkipDeepEquilibriumNetwork(main_layers, mapping_layers, nothing,
            nothing, solver, scales; sensealg, verbose=false, save_everystep=true)

        ps, st = Lux.setup(rng, model)
        x = randn(rng, Float32, 4, 1)

        gs = Zygote.gradient(loss_function, model, x, ps, st)

        @test is_finite_gradient(gs[2])
        @test is_finite_gradient(gs[3])

        st = Lux.update_state(st, :fixed_depth, Val(10))

        gs = Zygote.gradient(loss_function, model, x, ps, st)

        @test is_finite_gradient(gs[2])
        @test is_finite_gradient(gs[3])
    end

    return nothing
end

function test_multiscale_neural_ode_adjoint()
    rng = get_prng(0)

    solver = OrdinaryDiffEq.VCABM3()

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
    x = randn(rng, Float32, 4, 1)

    gs = Zygote.gradient(loss_function, model, x, ps, st)

    @test is_finite_gradient(gs[2])
    @test is_finite_gradient(gs[3])

    st = Lux.update_state(st, :fixed_depth, Val(10))

    gs = Zygote.gradient(loss_function, model, x, ps, st)

    @test is_finite_gradient(gs[2])
    @test is_finite_gradient(gs[3])

    return nothing
end

@testset "DeepEquilibriumNetwork" begin
    test_deep_equilibrium_network_adjoint()
end
@testset "SkipDeepEquilibriumNetwork" begin
    test_skip_deep_equilibrium_network_adjoint()
end
@testset "SkipRegDeepEquilibriumNetwork" begin
    test_skip_reg_deep_equilibrium_network_adjoint()
end
@testset "MultiScaleDeepEquilibriumNetwork" begin
    test_multiscale_deep_equilibrium_network_adjoint()
end
@testset "MultiScaleSkipDeepEquilibriumNetwork" begin
    test_multiscale_skip_deep_equilibrium_network_adjoint()
end
@testset "MultiScaleSkipDeepEquilibriumNetwork" begin
    test_multiscale_skip_reg_deep_equilibrium_network_adjoint()
end
@testset "MultiScaleNeuralODE" begin
    test_multiscale_neural_ode_adjoint()
end
