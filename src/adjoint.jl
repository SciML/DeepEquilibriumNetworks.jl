import ChainRulesCore as CRC
import DiffEqBase
import Functors
import LinearSolve
import SciMLBase
import SciMLSensitivity
import UnPack
import Zygote

@generated neg(x::T) where {T} = hasmethod(-, (T,)) ? :(-x) : :(x)
neg(nt::NamedTuple) = Functors.fmap(neg, nt)

@noinline function SteadyStateAdjointProblem(sol::EquilibriumSolution,
                                             sensealg::DeepEquilibriumAdjoint, alg,
                                             dgdu::DG1=nothing, dgdp::DG2=nothing,
                                             g::G=nothing; kwargs...) where {DG1, DG2, G}
  UnPack.@unpack f, p, u0 = sol.prob

  if dgdu === nothing && dgdp === nothing && g === nothing
    throw(ArgumentError("Either `dgdu`, `dgdp`, or `g` must be specified."))
  end

  diffcache, y = SciMLSensitivity.adjointdiffcache(g, sensealg, false, sol, dgdu, dgdp, f,
                                                   alg; quad=false, needs_jac=false,
                                                   noiseterm=false)

  if dgdp === nothing && g === nothing
    dgdu_val = diffcache.dg_val
    dgdp_val = nothing
  else
    dgdu_val, dgdp_val = diffcache.dg_val
  end

  if dgdu !== nothing
    dgdu(dgdu_val, y, p, nothing, nothing)
  else
    if g !== nothing
      if dgdp_val !== nothing
        SciMLSensitivity.gradient!(vec(dgdu_val), diffcache.g[1], y, sensealg,
                                   diffcache.g_grad_config[1])
      else
        SciMLSensitivity.gradient!(vec(dgdu_val), diffcache.g, y, sensealg,
                                   diffcache.g_grad_config)
      end
    end
  end

  if _check_adjoint_mode(sensealg, Val(:vanilla))
    # Solve the Linear Problem
    _val, back = Zygote.pullback(x -> f(x, p, nothing), y)
    s_val = size(_val)
    op = ZygotePullbackMultiplyOperator{eltype(y), typeof(back), typeof(s_val)}(back, s_val)
    linear_problem = LinearSolve.LinearProblem(op, vec(diffcache.dg_val))
    res = SciMLBase.solve(linear_problem, sensealg.linsolve).u
  elseif _check_adjoint_mode(sensealg, Val(:jfb))
    # Jacobian Free Backpropagation
    res = diffcache.dg_val
  else
    throw(ArgumentError("Unknown adjoint mode"))
  end

  # Compute the VJP
  _, back = Zygote.pullback(p -> vec(f(y, p, nothing)), p)
  dp = back(vec(res))[1]

  return neg(dp)
end

function DiffEqBase._concrete_solve_adjoint(prob::DiffEqBase.SteadyStateProblem, alg,
                                            sensealg::DeepEquilibriumAdjoint, u0, p,
                                            originator::SciMLBase.ADOriginator, args...;
                                            save_idxs=nothing, kwargs...)
  _prob = SciMLBase.remake(prob; u0=u0, p=p)
  sol = SciMLBase.solve(_prob, alg, args...; kwargs...)
  _save_idxs = save_idxs === nothing ? Colon() : save_idxs

  out = save_idxs === nothing ? sol : DiffEqBase.sensitivity_solution(sol, sol[_save_idxs])

  function steadystatebackpass(Δ)
    function df(_out, u, p, t, i)
      if typeof(_save_idxs) <: Number
        _out[_save_idxs] = Δ[_save_idxs]
      elseif typeof(Δ) <: Number
        _out[_save_idxs] .= Δ
      else
        _out[_save_idxs] .= Δ[_save_idxs]
      end
    end

    dp = SciMLSensitivity.adjoint_sensitivities(sol, alg; sensealg=sensealg, dgdu=df)

    if originator isa SciMLBase.TrackerOriginator ||
       originator isa SciMLBase.ReverseDiffOriginator
      return (CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), dp, CRC.NoTangent(),
              ntuple(_ -> CRC.NoTangent(), length(args))...)
    else
      return (CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), dp,
              CRC.NoTangent(), ntuple(_ -> CRC.NoTangent(), length(args))...)
    end
  end

  return out, steadystatebackpass
end

function SciMLSensitivity._adjoint_sensitivities(sol, sensealg::DeepEquilibriumAdjoint, alg;
                                                 g=nothing, dgdu=nothing, dgdp=nothing,
                                                 abstol=1e-6, reltol=1e-3, kwargs...)
  return SteadyStateAdjointProblem(sol, sensealg, alg, dgdu, dgdp, g; kwargs...)
end
