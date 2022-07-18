import DeepEquilibriumNetworks as DEQs
import Lux
import OrdinaryDiffEq
import Test

include("../test_utils.jl")

function test_deep_equilibrium_network()
  rng = get_prng(0)

  continuous_solver = DEQs.ContinuousDEQSolver(OrdinaryDiffEq.VCABM3(); abstol=0.01f0,
                                               reltol=0.01f0, abstol_termination=0.01f0,
                                               reltol_termination=0.01f0)
  discrete_solver = DEQs.DiscreteDEQSolver(DEQs.LimitedMemoryBroydenSolver();
                                           abstol_termination=0.01f0,
                                           reltol_termination=0.01f0)

  for solver in (continuous_solver, discrete_solver)
    model = DEQs.DeepEquilibriumNetwork(Lux.Parallel(+, get_dense_layer(2, 2; bias=false),
                                                     get_dense_layer(2, 2; bias=false)),
                                        solver; verbose=false)

    ps, st = Lux.setup(rng, model)

    Test.@test st.solution === nothing

    x = randn(rng, Float32, 2, 1)

    z, st = model(x, ps, st)

    Test.@test all(isfinite, z)
    Test.@test size(z) == size(x)
    Test.@test st.solution isa DEQs.DeepEquilibriumSolution

    ps, st = Lux.setup(rng, model)
    st = Lux.update_state(st, :fixed_depth, Val(10))

    Test.@test st.solution === nothing

    z, st = model(x, ps, st)

    Test.@test all(isfinite, z)
    Test.@test size(z) == size(x)
    Test.@test st.solution isa DEQs.DeepEquilibriumSolution
    Test.@test DEQs.number_of_function_evaluations(st.solution) == 10
  end

  return nothing
end

function test_skip_deep_equilibrium_network()
  rng = get_prng(0)

  continuous_solver = DEQs.ContinuousDEQSolver(OrdinaryDiffEq.VCABM3(); abstol=0.01f0,
                                               reltol=0.01f0, abstol_termination=0.01f0,
                                               reltol_termination=0.01f0)
  discrete_solver = DEQs.DiscreteDEQSolver(DEQs.LimitedMemoryBroydenSolver();
                                           abstol_termination=0.01f0,
                                           reltol_termination=0.01f0)

  for solver in (continuous_solver, discrete_solver)
    model = DEQs.SkipDeepEquilibriumNetwork(Lux.Parallel(+,
                                                         get_dense_layer(2, 2; bias=false),
                                                         get_dense_layer(2, 2; bias=false)),
                                            get_dense_layer(2, 2), solver; verbose=false)

    ps, st = Lux.setup(rng, model)

    Test.@test st.solution === nothing

    x = randn(rng, Float32, 2, 1)

    z, st = model(x, ps, st)

    Test.@test all(isfinite, z)
    Test.@test size(z) == size(x)
    Test.@test st.solution isa DEQs.DeepEquilibriumSolution

    ps, st = Lux.setup(rng, model)
    st = Lux.update_state(st, :fixed_depth, Val(10))

    Test.@test st.solution === nothing

    z, st = model(x, ps, st)

    Test.@test all(isfinite, z)
    Test.@test size(z) == size(x)
    Test.@test st.solution isa DEQs.DeepEquilibriumSolution
    Test.@test DEQs.number_of_function_evaluations(st.solution) == 10
  end

  return nothing
end

function test_skip_deep_equilibrium_network_v2()
  rng = get_prng(0)

  continuous_solver = DEQs.ContinuousDEQSolver(OrdinaryDiffEq.VCABM3(); abstol=0.01f0,
                                               reltol=0.01f0, abstol_termination=0.01f0,
                                               reltol_termination=0.01f0)
  discrete_solver = DEQs.DiscreteDEQSolver(DEQs.LimitedMemoryBroydenSolver();
                                           abstol_termination=0.01f0,
                                           reltol_termination=0.01f0)

  for solver in (continuous_solver, discrete_solver)
    model = DEQs.SkipDeepEquilibriumNetwork(Lux.Parallel(+,
                                                         get_dense_layer(2, 2; bias=false),
                                                         get_dense_layer(2, 2; bias=false)),
                                            nothing, solver; verbose=false)

    ps, st = Lux.setup(rng, model)

    Test.@test st.solution === nothing

    x = randn(rng, Float32, 2, 1)

    z, st = model(x, ps, st)

    Test.@test all(isfinite, z)
    Test.@test size(z) == size(x)
    Test.@test st.solution isa DEQs.DeepEquilibriumSolution

    ps, st = Lux.setup(rng, model)
    st = Lux.update_state(st, :fixed_depth, Val(10))

    Test.@test st.solution === nothing

    z, st = model(x, ps, st)

    Test.@test all(isfinite, z)
    Test.@test size(z) == size(x)
    Test.@test st.solution isa DEQs.DeepEquilibriumSolution
    Test.@test DEQs.number_of_function_evaluations(st.solution) == 10
  end

  return nothing
end

Test.@testset "DeepEquilibriumNetwork" begin test_deep_equilibrium_network() end
Test.@testset "SkipDeepEquilibriumNetwork" begin test_skip_deep_equilibrium_network() end
Test.@testset "SkipDeepEquilibriumNetworkV2" begin test_skip_deep_equilibrium_network_v2() end
