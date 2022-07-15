import DeepEquilibriumNetworks as DEQs
import Lux
import Test
import Zygote

include("test_utils.jl")

function test_zygote_pullback_multiply_operator()
  rng = get_prng(0)
  model = Lux.Chain(Lux.Dense(4 => 5, tanh), Lux.Dense(5 => 2))
  ps, st = Lux.setup(rng, model)

  x = randn(rng, Float32, 4, 1)
  v = randn(rng, Float32, 2)

  _val, pullback = Zygote.pullback(x -> model(x, ps, st)[1], x)
  s_val = size(_val)
  op = DEQs.ZygotePullbackMultiplyOperator{eltype(x), typeof(pullback), typeof(s_val)}(pullback,
                                                                                       s_val)

  Test.@test size(op * v) == size(x)

  return nothing
end

Test.@testset "ZygotePullbackMultiplyOperator" begin test_zygote_pullback_multiply_operator() end
