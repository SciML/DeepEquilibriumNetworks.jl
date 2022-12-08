import DeepEquilibriumNetworks as DEQs
import Lux
import OrdinaryDiffEq
import Test

include("../test_utils.jl")

function test_multiscale_neural_ode()
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
  ps = Lux.ComponentArray(ps)

  Test.@test st.solution === nothing

  x = randn(rng, Float32, 4, 1)

  z, st = model(x, ps, st)

  Test.@test all(Base.Fix1(all, isfinite), z)
  Test.@test all(map(x -> size(x)[1:(end - 1)], z) .== scales)
  Test.@test st.solution isa DEQs.DeepEquilibriumSolution

  ps, st = Lux.setup(rng, model)
  st = Lux.update_state(st, :fixed_depth, Val(10))

  Test.@test st.solution === nothing

  z, st = model(x, ps, st)

  Test.@test all(Base.Fix1(all, isfinite), z)
  Test.@test all(map(x -> size(x)[1:(end - 1)], z) .== scales)
  Test.@test st.solution isa DEQs.DeepEquilibriumSolution
  Test.@test DEQs.number_of_function_evaluations(st.solution) == 10

  return nothing
end

test_multiscale_neural_ode()
