import DeepEquilibriumNetworks as DEQs
import Functors
import JET
import Lux
import Random

function get_prng(seed::Int)
  @static if VERSION >= v"1.7"
    rng = Random.Xoshiro()
    Random.seed!(rng, seed)
    return rng
  else
    rng = Random.MersenneTwister()
    Random.seed!(rng, seed)
    return rng
  end
end

is_finite_gradient(x::AbstractArray) = all(isfinite, x)

function is_finite_gradient(gs::NamedTuple)
  gradient_is_finite = [true]
  function _is_gradient_finite(x)
    if !isnothing(x) && !all(isfinite, x)
      gradient_is_finite[1] = false
    end
    return x
  end
  Functors.fmap(_is_gradient_finite, gs)
  return gradient_is_finite[1]
end

function run_JET_tests(f, args...; call_broken=false, opt_broken=false, kwargs...)
  @static if VERSION >= v"1.7"
    JET.test_call(f, typeof.(args); broken=call_broken)
    JET.test_opt(f, typeof.(args); broken=opt_broken, target_modules=(DEQs,))
  end
end

function get_dense_layer(args...; kwargs...)
  function init_weight(rng::Random.AbstractRNG, out_dims, in_dims)
    return randn(rng, Float32, (out_dims, in_dims)) .* 0.001f0
  end
  return Lux.Dense(args...; init_weight, kwargs...)
end
