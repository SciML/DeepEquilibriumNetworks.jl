using DeepEquilibriumNetworks, Lux, Zygote
using Test

include("test_utils.jl")

function test_zygote_pullback_multiply_operator()
  rng = get_prng(0)
  model = Chain(Dense(4 => 5, tanh), Dense(5 => 2))
  ps, st = Lux.setup(rng, model)

  x = randn(rng, Float32, 4, 1)
  v = randn(rng, Float32, 2)

  _val, pullback = Zygote.pullback(x -> model(x, ps, st)[1], x)
  op = DEQs._zygote_pullback_operator(pullback, _val)

  @test size(op * v) == size(x)

  return nothing
end

@testset "ZygotePullbackMultiplyOperator" begin test_zygote_pullback_multiply_operator() end
