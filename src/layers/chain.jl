"""
    DEQChain(layers...)

Sequence of layers divided into 3 chunks --

  - `pre_deq` -- layers that are executed before DEQ is applied
  - `deq` -- The Deep Equilibrium Layer
  - `post_deq` -- layers that are executed after DEQ is applied

Constraint: Must have one DEQ layer in `layers`
"""
struct DEQChain{P1, D, P2} <: AbstractExplicitContainerLayer{(:pre_deq, :deq, :post_deq)}
  pre_deq::P1
  deq::D
  post_deq::P2
end

function DEQChain(layers...)
  pre_deq, post_deq, deq, encounter_deq = [], [], nothing, false
  for l in layers
    if l isa AbstractDeepEquilibriumNetwork || l isa AbstractSkipDeepEquilibriumNetwork
      @assert !encounter_deq "Can have only 1 DEQ Layer in the Chain!!!"
      deq = l
      encounter_deq = true
      continue
    end
    push!(encounter_deq ? post_deq : pre_deq, l)
  end
  @assert encounter_deq "No DEQ Layer in the Chain!!! Maybe you wanted to use Chain"
  pre_deq = length(pre_deq) == 0 ? NoOpLayer() : Chain(pre_deq...)
  post_deq = length(post_deq) == 0 ? NoOpLayer() : Chain(post_deq...)
  return DEQChain(pre_deq, deq, post_deq)
end

function get_deq_return_type(deq::DEQChain{P1,
                                           <:Union{MultiScaleDeepEquilibriumNetwork,
                                                   MultiScaleSkipDeepEquilibriumNetwork}},
                             ::T) where {P1, T}
  return NTuple{length(deq.deq.scales), T}
end
get_deq_return_type(::DEQChain, ::T) where {T} = T

function (deq::DEQChain)(x, ps::Union{ComponentArray, NamedTuple}, st::NamedTuple)
  T = get_deq_return_type(deq, x)
  x1, st1 = deq.pre_deq(x, ps.pre_deq, st.pre_deq)
  (x2::T, deq_soln), st2 = deq.deq(x1, ps.deq, st.deq)
  x3, st3 = deq.post_deq(x2, ps.post_deq, st.post_deq)
  return (x3, deq_soln), (pre_deq=st1, deq=st2, post_deq=st3)
end
