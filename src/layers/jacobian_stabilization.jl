_gaussian_like(rng::AbstractRNG, x::AbstractArray) = randn(rng, eltype(x), size(x))
_gaussian_like(rng::AbstractRNG, x::CuArray) = CUDA.randn(rng, eltype(x), size(x))

"""
    estimate_jacobian_trace(::Val{mode}, model::Lux.AbstractExplicitLayer, ps,
                            st::NamedTuple, z::AbstractArray, x::AbstractArray,
                            rng::Random.AbstractRNG)

Estimates the trace of the jacobian matrix wrt `z`.

## Arguments

  - `mode`: Options: `reverse` and `finite_diff`
  - `model`: A `Lux` Neural Network mapping 2 equal sized arrays to a same sized array. This
    convention is not checked, and if violated will lead to errors.
  - `ps`: Parameters of `model`.
  - `st`: States of `model`.
  - `z`: Input wrt the Jacobian is computed.
  - `x`: Other Input to `model`.
  - `rng`: PRNG. Note that this object is mutated by this function.

## Returns

Stochastic Estimate of the trace of the Jacobian.
"""
function estimate_jacobian_trace(::Val{:reverse},
  model::Lux.AbstractExplicitLayer,
  ps,
  st::NamedTuple,
  z::AbstractArray,
  x::AbstractArray,
  rng::AbstractRNG)
  _, back = Zygote.pullback(u -> model((u, x), ps, st)[1], z)
  vjp_z = back(_gaussian_like(rng, x))[1]
  return mean(abs2, vjp_z)
end

function estimate_jacobian_trace(::Val{:finite_diff},
  model::Lux.AbstractExplicitLayer,
  ps,
  st::NamedTuple,
  z::AbstractArray,
  x::AbstractArray,
  rng::AbstractRNG)
  f = u -> model((u, x), ps, st)[1]
  res = convert(eltype(z), 0)
  epsilon = cbrt(eps(typeof(res)))
  _epsilon = inv(epsilon)
  f0 = f(z)
  v = _gaussian_like(rng, x)

  for idx in eachindex(z)
    _z = z[idx]
    CRC.ignore_derivatives() do
      return z[idx] = z[idx] + epsilon
    end
    res = res + abs2(sum((f(z) .- f0) .* _epsilon .* v)) / length(v)
    CRC.ignore_derivatives() do
      return z[idx] = _z
    end
  end
  return res
end
