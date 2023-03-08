const SingleScaleDeepEquilibriumNetworks = Union{DeepEquilibriumNetwork,
                                                 SkipDeepEquilibriumNetwork}

function (deq::SingleScaleDeepEquilibriumNetworks)(x::AbstractArray{T}, ps, st::NamedTuple,
                                                   ::Val{true}) where {T}
  # Pretraining without Fixed Point Solving
  z, st = _get_initial_condition(deq, x, ps, st)
  depth = _get_unrolled_depth(st)

  z_star, st_ = _evaluate_unrolled_model(deq, deq.model, z, x, ps.model, st.model,
                                         st.fixed_depth)

  residual = CRC.ignore_derivatives(z_star .- deq.model((z_star, x), ps.model, st.model)[1])

  @set! st.model = st_
  @set! st.solution = DeepEquilibriumSolution(z_star, z, residual, T(0), depth)

  return z_star, st
end

function (deq::SingleScaleDeepEquilibriumNetworks)(x::AbstractArray{T}, ps, st::NamedTuple,
                                                   ::Val{false}) where {T}
  z, st = _get_initial_condition(deq, x, ps, st)
  st_, nfe = st.model, 0

  function dudt(u, p, t)
    nfe += 1
    u_, st_ = deq.model((u, x), p, st_)
    return u_ .- u
  end

  prob = SteadyStateProblem(ODEFunction{false}(dudt), z, ps.model)
  sol = solve(prob, deq.solver; deq.sensealg, deq.kwargs...)

  z_star, st_ = deq.model((sol.u, x), ps.model, st_)

  # if _jacobian_regularization(deq)
  #   rng = Lux.replicate(st.rng)
  #   jac_loss = estimate_jacobian_trace(Val(:finite_diff), deq.model, ps, st.model, z_star,
  #                                      x, rng)
  # else
  #   rng = st.rng
  #   jac_loss = T(0)
  # end
  jac_loss = T(0)

  residual = CRC.ignore_derivatives(z_star .- deq.model((z_star, x), ps.model, st.model)[1])

  @set! st.model = st_
  # @set! st.rng = rng
  @set! st.solution = DeepEquilibriumSolution(z_star, z, residual, jac_loss, nfe)

  return z_star, st
end