import DeepEquilibriumNetworks as DEQs
import Lux
import NNlib
import Random
import Test
import Zygote

include("../test_utils.jl")

function test_jacobian_trace_estimation()
  rng = get_prng(0)
  model = Lux.Parallel(+, get_dense_layer(3 => 3), get_dense_layer(3 => 3))
  ps, st = Lux.setup(rng, model)

  z = randn(rng, Float32, 3, 2)
  x = randn(rng, Float32, 3, 2)

  reverse_mode_estimate = DEQs.estimate_jacobian_trace(Val(:reverse_mode), model, ps, st, z,
                                                       x, Lux.replicate(rng))
  run_JET_tests(DEQs.estimate_jacobian_trace, Val(:reverse_mode), model, ps, st, z, x,
                Lux.replicate(rng); call_broken=true)

  finite_diff_estimate = DEQs.estimate_jacobian_trace(Val(:finite_diff), model, ps, st, z,
                                                      x, Lux.replicate(rng))
  run_JET_tests(DEQs.estimate_jacobian_trace, Val(:finite_diff), model, ps, st, z, x,
                Lux.replicate(rng))

  Test.@test isapprox(reverse_mode_estimate, finite_diff_estimate)

  model = Lux.Parallel(+, Lux.Conv((1, 1), 3 => 3, NNlib.relu), Lux.Conv((1, 1), 3 => 3))
  ps, st = Lux.setup(rng, model)

  z = randn(rng, Float32, 5, 5, 3, 2)
  x = randn(rng, Float32, 5, 5, 3, 2)

  reverse_mode_estimate = DEQs.estimate_jacobian_trace(Val(:reverse_mode), model, ps, st, z,
                                                       x, Lux.replicate(rng))
  run_JET_tests(DEQs.estimate_jacobian_trace, Val(:reverse_mode), model, ps, st, z, x,
                Lux.replicate(rng); call_broken=true)

  finite_diff_estimate = DEQs.estimate_jacobian_trace(Val(:finite_diff), model, ps, st, z,
                                                      x, Lux.replicate(rng))
  run_JET_tests(DEQs.estimate_jacobian_trace, Val(:finite_diff), model, ps, st, z, x,
                Lux.replicate(rng))

  Test.@test isapprox(reverse_mode_estimate, finite_diff_estimate)

  return nothing
end

function test_jacobian_trace_estimation_gradient()
  rng = get_prng(0)
  model = Lux.Parallel(+, get_dense_layer(3 => 3), get_dense_layer(3 => 3))
  ps, st = Lux.setup(rng, model)

  z = randn(rng, Float32, 3, 2)
  x = randn(rng, Float32, 3, 2)

  gs = Zygote.gradient(ps -> DEQs.estimate_jacobian_trace(Val(:finite_diff), model, ps, st,
                                                          z, x, Lux.replicate(rng)), ps)[1]

  Test.@test is_finite_gradient(gs)

  model = Lux.Parallel(+, Lux.Conv((1, 1), 3 => 3, NNlib.relu), Lux.Conv((1, 1), 3 => 3))
  ps, st = Lux.setup(rng, model)

  z = randn(rng, Float32, 5, 5, 3, 2)
  x = randn(rng, Float32, 5, 5, 3, 2)

  gs = Zygote.gradient(ps -> DEQs.estimate_jacobian_trace(Val(:finite_diff), model, ps, st,
                                                          z, x, Lux.replicate(rng)), ps)[1]

  Test.@test is_finite_gradient(gs)

  return nothing
end

Test.@testset "Jacobian Trace Extimation" begin test_jacobian_trace_estimation() end
Test.@testset "Jacobian Trace Extimation: Gradient" begin test_jacobian_trace_estimation_gradient() end
