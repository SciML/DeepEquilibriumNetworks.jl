import DeepEquilibriumNetworks as DEQs
import ComponentArrays as CA
import Lux
import OrdinaryDiffEq
import Test
import Zygote

include("test_utils.jl")

function loss_function(model::DEQs.AbstractDeepEquilibriumNetwork, x, ps, st)
  y, st_ = model(x, ps, st)
  return sum(y) + DEQs.jacobian_loss(st_.solution)
end

function loss_function(model::DEQs.AbstractSkipDeepEquilibriumNetwork, x, ps, st)
  y, st_ = model(x, ps, st)
  return sum(y) + DEQs.jacobian_loss(st_.solution) + DEQs.skip_loss(st_.solution)
end

function loss_function(model::Union{DEQs.MultiScaleDeepEquilibriumNetwork,
                                    DEQs.MultiScaleNeuralODE}, x, ps, st)
  y, st_ = model(x, ps, st)
  return sum(sum, y) + DEQs.jacobian_loss(st_.solution)
end

function loss_function(model::DEQs.MultiScaleSkipDeepEquilibriumNetwork, x, ps, st)
  y, st_ = model(x, ps, st)
  return sum(sum, y) + DEQs.jacobian_loss(st_.solution) + DEQs.skip_loss(st_.solution)
end

function test_zygote_pullback_multiply_operator()
  rng = get_prng(0)
  model = Lux.Chain(Lux.Dense(4 => 5, tanh), Lux.Dense(5 => 2))
  ps, st = Lux.setup(rng, model)

  x = randn(rng, Float32, 4, 1)
  v = randn(rng, Float32, 2)

  _val, pullback = Zygote.pullback(x -> model(x, ps, st)[1], x)
  op = DEQs._zygote_pullback_operator(pullback, _val)

  Test.@test size(op * v) == size(x)

  return nothing
end

function test_deep_equilibrium_network_adjoint()
  rng = get_prng(0)

  continuous_solver = DEQs.ContinuousDEQSolver(OrdinaryDiffEq.VCABM3(); abstol=0.01f0,
                                               reltol=0.01f0, abstol_termination=0.01f0,
                                               reltol_termination=0.01f0)
  discrete_solver = DEQs.DiscreteDEQSolver(DEQs.LimitedMemoryBroydenSolver();
                                           abstol_termination=0.01f0,
                                           reltol_termination=0.01f0)

  for solver in (continuous_solver, discrete_solver),
      mode in (:vanilla, :jfb),
      jacobian_regularization in (true, false)

    sensealg = DEQs.DeepEquilibriumAdjoint(0.01f0, 0.01f0, 10; mode)
    model = DEQs.DeepEquilibriumNetwork(Lux.Parallel(+, get_dense_layer(2, 2; bias=false),
                                                     get_dense_layer(2, 2; bias=false)),
                                        solver; sensealg, jacobian_regularization,
                                        verbose=false)

    ps, st = Lux.setup(rng, model)
    x = randn(rng, Float32, 2, 1)

    gs = Zygote.gradient((x, ps) -> loss_function(model, x, ps, st), x, ps)

    Test.@test is_finite_gradient(gs[1])
    Test.@test is_finite_gradient(gs[2])

    st = Lux.update_state(st, :fixed_depth, Val(10))

    gs = Zygote.gradient((x, ps) -> loss_function(model, x, ps, st), x, ps)

    Test.@test is_finite_gradient(gs[1])
    Test.@test is_finite_gradient(gs[2])
  end

  return nothing
end

function test_skip_deep_equilibrium_network_adjoint()
  rng = get_prng(0)

  continuous_solver = DEQs.ContinuousDEQSolver(OrdinaryDiffEq.VCABM3(); abstol=0.01f0,
                                               reltol=0.01f0, abstol_termination=0.01f0,
                                               reltol_termination=0.01f0)
  discrete_solver = DEQs.DiscreteDEQSolver(DEQs.LimitedMemoryBroydenSolver();
                                           abstol_termination=0.01f0,
                                           reltol_termination=0.01f0)

  for solver in (continuous_solver, discrete_solver),
      mode in (:vanilla, :jfb),
      jacobian_regularization in (true, false)

    sensealg = DEQs.DeepEquilibriumAdjoint(0.01f0, 0.01f0, 10; mode)
    model = DEQs.SkipDeepEquilibriumNetwork(Lux.Parallel(+,
                                                         get_dense_layer(2, 2; bias=false),
                                                         get_dense_layer(2, 2; bias=false)),
                                            get_dense_layer(2, 2), solver; sensealg,
                                            jacobian_regularization, verbose=false)

    ps, st = Lux.setup(rng, model)
    x = randn(rng, Float32, 2, 1)

    gs = Zygote.gradient((x, ps) -> loss_function(model, x, ps, st), x, ps)

    Test.@test is_finite_gradient(gs[1])
    Test.@test is_finite_gradient(gs[2])

    st = Lux.update_state(st, :fixed_depth, Val(10))

    gs = Zygote.gradient((x, ps) -> loss_function(model, x, ps, st), x, ps)

    Test.@test is_finite_gradient(gs[1])
    Test.@test is_finite_gradient(gs[2])
  end

  return nothing
end

function test_skip_deep_equilibrium_network_v2_adjoint()
  rng = get_prng(0)

  continuous_solver = DEQs.ContinuousDEQSolver(OrdinaryDiffEq.VCABM3(); abstol=0.01f0,
                                               reltol=0.01f0, abstol_termination=0.01f0,
                                               reltol_termination=0.01f0)
  discrete_solver = DEQs.DiscreteDEQSolver(DEQs.LimitedMemoryBroydenSolver();
                                           abstol_termination=0.01f0,
                                           reltol_termination=0.01f0)

  for solver in (continuous_solver, discrete_solver),
      mode in (:vanilla, :jfb),
      jacobian_regularization in (true, false)

    sensealg = DEQs.DeepEquilibriumAdjoint(0.01f0, 0.01f0, 10; mode)
    model = DEQs.SkipDeepEquilibriumNetwork(Lux.Parallel(+,
                                                         get_dense_layer(2, 2; bias=false),
                                                         get_dense_layer(2, 2; bias=false)),
                                            nothing, solver; sensealg,
                                            jacobian_regularization, verbose=false)

    ps, st = Lux.setup(rng, model)
    x = randn(rng, Float32, 2, 1)

    gs = Zygote.gradient((x, ps) -> loss_function(model, x, ps, st), x, ps)

    Test.@test is_finite_gradient(gs[1])
    Test.@test is_finite_gradient(gs[2])

    st = Lux.update_state(st, :fixed_depth, Val(10))

    gs = Zygote.gradient((x, ps) -> loss_function(model, x, ps, st), x, ps)

    Test.@test is_finite_gradient(gs[1])
    Test.@test is_finite_gradient(gs[2])
  end

  return nothing
end

function test_multiscale_deep_equilibrium_network_adjoint()
  rng = get_prng(0)

  continuous_solver = DEQs.ContinuousDEQSolver(OrdinaryDiffEq.VCABM3(); abstol=0.01f0,
                                               reltol=0.01f0, abstol_termination=0.01f0,
                                               reltol_termination=0.01f0)
  discrete_solver = DEQs.DiscreteDEQSolver(DEQs.LimitedMemoryBroydenSolver();
                                           abstol_termination=0.01f0,
                                           reltol_termination=0.01f0)

  main_layers = (Lux.Parallel(+, get_dense_layer(4, 4), get_dense_layer(4, 4)),
                 get_dense_layer(3, 3), get_dense_layer(2, 2), get_dense_layer(1, 1))

  mapping_layers = [Lux.NoOpLayer() get_dense_layer(4, 3) get_dense_layer(4, 2) get_dense_layer(4, 1);
                    get_dense_layer(3, 4) Lux.NoOpLayer() get_dense_layer(3, 2) get_dense_layer(3, 1);
                    get_dense_layer(2, 4) get_dense_layer(2, 3) Lux.NoOpLayer() get_dense_layer(2, 1);
                    get_dense_layer(1, 4) get_dense_layer(1, 3) get_dense_layer(1, 2) Lux.NoOpLayer()]

  for solver in (continuous_solver, discrete_solver), mode in (:vanilla, :jfb)
    sensealg = DEQs.DeepEquilibriumAdjoint(0.01f0, 0.01f0, 10; mode)
    scales = ((4,), (3,), (2,), (1,))
    model = DEQs.MultiScaleDeepEquilibriumNetwork(main_layers, mapping_layers, nothing,
                                                  solver, scales; sensealg, verbose=false)

    ps, st = Lux.setup(rng, model)

    x = randn(rng, Float32, 4, 1)

    gs = Zygote.gradient((x, ps) -> loss_function(model, x, ps, st), x, ps)

    Test.@test is_finite_gradient(gs[1])
    Test.@test is_finite_gradient(gs[2])

    st = Lux.update_state(st, :fixed_depth, Val(10))

    gs = Zygote.gradient((x, ps) -> loss_function(model, x, ps, st), x, ps)

    Test.@test is_finite_gradient(gs[1])
    Test.@test is_finite_gradient(gs[2])
  end

  return nothing
end

function test_multiscale_skip_deep_equilibrium_network_adjoint()
  rng = get_prng(0)

  continuous_solver = DEQs.ContinuousDEQSolver(OrdinaryDiffEq.VCABM3(); abstol=0.01f0,
                                               reltol=0.01f0, abstol_termination=0.01f0,
                                               reltol_termination=0.01f0)
  discrete_solver = DEQs.DiscreteDEQSolver(DEQs.LimitedMemoryBroydenSolver();
                                           abstol_termination=0.01f0,
                                           reltol_termination=0.01f0)

  main_layers = (Lux.Parallel(+, get_dense_layer(4, 4), get_dense_layer(4, 4)),
                 get_dense_layer(3, 3), get_dense_layer(2, 2), get_dense_layer(1, 1))

  mapping_layers = [Lux.NoOpLayer() get_dense_layer(4, 3) get_dense_layer(4, 2) get_dense_layer(4, 1);
                    get_dense_layer(3, 4) Lux.NoOpLayer() get_dense_layer(3, 2) get_dense_layer(3, 1);
                    get_dense_layer(2, 4) get_dense_layer(2, 3) Lux.NoOpLayer() get_dense_layer(2, 1);
                    get_dense_layer(1, 4) get_dense_layer(1, 3) get_dense_layer(1, 2) Lux.NoOpLayer()]

  shortcut_layers = (get_dense_layer(4, 4), get_dense_layer(4, 3), get_dense_layer(4, 2),
                     get_dense_layer(4, 1))

  for solver in (continuous_solver, discrete_solver), mode in (:vanilla, :jfb)
    sensealg = DEQs.DeepEquilibriumAdjoint(0.01f0, 0.01f0, 10; mode)
    scales = ((4,), (3,), (2,), (1,))
    model = DEQs.MultiScaleSkipDeepEquilibriumNetwork(main_layers, mapping_layers, nothing,
                                                      shortcut_layers, solver, scales;
                                                      sensealg, verbose=false)

    ps, st = Lux.setup(rng, model)

    x = randn(rng, Float32, 4, 1)

    gs = Zygote.gradient((x, ps) -> loss_function(model, x, ps, st), x, ps)

    Test.@test is_finite_gradient(gs[1])
    Test.@test is_finite_gradient(gs[2])

    st = Lux.update_state(st, :fixed_depth, Val(10))

    gs = Zygote.gradient((x, ps) -> loss_function(model, x, ps, st), x, ps)

    Test.@test is_finite_gradient(gs[1])
    Test.@test is_finite_gradient(gs[2])
  end

  return nothing
end

function test_multiscale_skip_deep_equilibrium_network_v2_adjoint()
  rng = get_prng(0)

  continuous_solver = DEQs.ContinuousDEQSolver(OrdinaryDiffEq.VCABM3(); abstol=0.01f0,
                                               reltol=0.01f0, abstol_termination=0.01f0,
                                               reltol_termination=0.01f0)
  discrete_solver = DEQs.DiscreteDEQSolver(DEQs.LimitedMemoryBroydenSolver();
                                           abstol_termination=0.01f0,
                                           reltol_termination=0.01f0)

  main_layers = (Lux.Parallel(+, get_dense_layer(4, 4), get_dense_layer(4, 4)),
                 get_dense_layer(3, 3), get_dense_layer(2, 2), get_dense_layer(1, 1))

  mapping_layers = [Lux.NoOpLayer() get_dense_layer(4, 3) get_dense_layer(4, 2) get_dense_layer(4, 1);
                    get_dense_layer(3, 4) Lux.NoOpLayer() get_dense_layer(3, 2) get_dense_layer(3, 1);
                    get_dense_layer(2, 4) get_dense_layer(2, 3) Lux.NoOpLayer() get_dense_layer(2, 1);
                    get_dense_layer(1, 4) get_dense_layer(1, 3) get_dense_layer(1, 2) Lux.NoOpLayer()]

  for solver in (continuous_solver, discrete_solver), mode in (:vanilla, :jfb)
    sensealg = DEQs.DeepEquilibriumAdjoint(0.01f0, 0.01f0, 10; mode)
    scales = ((4,), (3,), (2,), (1,))
    model = DEQs.MultiScaleSkipDeepEquilibriumNetwork(main_layers, mapping_layers, nothing,
                                                      nothing, solver, scales; sensealg,
                                                      verbose=false)

    ps, st = Lux.setup(rng, model)

    x = randn(rng, Float32, 4, 1)

    gs = Zygote.gradient((x, ps) -> loss_function(model, x, ps, st), x, ps)

    Test.@test is_finite_gradient(gs[1])
    Test.@test is_finite_gradient(gs[2])

    st = Lux.update_state(st, :fixed_depth, Val(10))

    gs = Zygote.gradient((x, ps) -> loss_function(model, x, ps, st), x, ps)

    Test.@test is_finite_gradient(gs[1])
    Test.@test is_finite_gradient(gs[2])
  end

  return nothing
end

function test_multiscale_neural_ode_adjoint()
  rng = get_prng(0)

  solver = OrdinaryDiffEq.VCABM3()

  main_layers = (Lux.Parallel(+, get_dense_layer(4, 4), get_dense_layer(4, 4)),
                 get_dense_layer(3, 3), get_dense_layer(2, 2), get_dense_layer(1, 1))

  mapping_layers = [Lux.NoOpLayer() get_dense_layer(4, 3) get_dense_layer(4, 2) get_dense_layer(4, 1);
                    get_dense_layer(3, 4) Lux.NoOpLayer() get_dense_layer(3, 2) get_dense_layer(3, 1);
                    get_dense_layer(2, 4) get_dense_layer(2, 3) Lux.NoOpLayer() get_dense_layer(2, 1);
                    get_dense_layer(1, 4) get_dense_layer(1, 3) get_dense_layer(1, 2) Lux.NoOpLayer()]

  scales = ((4,), (3,), (2,), (1,))
  model = DEQs.MultiScaleNeuralODE(main_layers, mapping_layers, nothing, solver, scales;
                                   abstol=0.01f0, reltol=0.01f0)

  ps, st = Lux.setup(rng, model)
  ps = CA.ComponentArray(ps)

  x = randn(rng, Float32, 4, 1)

  gs = Zygote.gradient((x, ps) -> loss_function(model, x, ps, st), x, ps)

  Test.@test is_finite_gradient(gs[1])
  Test.@test is_finite_gradient(gs[2])

  st = Lux.update_state(st, :fixed_depth, Val(10))

  gs = Zygote.gradient((x, ps) -> loss_function(model, x, ps, st), x, ps)

  Test.@test is_finite_gradient(gs[1])
  Test.@test is_finite_gradient(gs[2])

  return nothing
end


Test.@testset "ZygotePullbackMultiplyOperator" begin test_zygote_pullback_multiply_operator() end
Test.@testset "DeepEquilibriumNetwork" begin test_deep_equilibrium_network_adjoint() end
Test.@testset "SkipDeepEquilibriumNetwork" begin test_skip_deep_equilibrium_network_adjoint() end
Test.@testset "SkipDeepEquilibriumNetworkV2" begin test_skip_deep_equilibrium_network_v2_adjoint() end
Test.@testset "MultiScaleDeepEquilibriumNetwork" begin test_multiscale_deep_equilibrium_network_adjoint() end
Test.@testset "MultiScaleSkipDeepEquilibriumNetwork" begin test_multiscale_skip_deep_equilibrium_network_adjoint() end
Test.@testset "MultiScaleSkipDeepEquilibriumNetworkV2" begin test_multiscale_skip_deep_equilibrium_network_v2_adjoint() end
Test.@testset "MultiScaleNeuralODE" begin test_multiscale_neural_ode_adjoint() end
