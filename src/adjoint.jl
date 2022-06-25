neg(x::Any) = hasmethod(-, (typeof(x),)) ? -x : x
neg(nt::NamedTuple) = fmap(neg, nt)

@noinline function SciMLSensitivity.SteadyStateAdjointProblem(sol::EquilibriumSolution,
                                                               sensealg::DeepEquilibriumAdjoint,
                                                               g::Nothing, dg;
                                                               save_idxs=nothing)
    @unpack f, p, u0 = sol.prob

    diffcache, y = SciMLSensitivity.adjointdiffcache(g, sensealg, false, sol, dg, f;
                                                      quad=false, needs_jac=false)

    _save_idxs = save_idxs === nothing ? Colon() : save_idxs
    if dg !== nothing
        if typeof(_save_idxs) <: Number
            diffcache.dg_val[_save_idxs] = dg[_save_idxs]
        elseif typeof(dg) <: Number
            @. diffcache.dg_val[_save_idxs] = dg
        else
            @. diffcache.dg_val[_save_idxs] = dg[_save_idxs]
        end
    end

    if check_adjoint_mode(sensealg, Val(:vanilla))
        # Solve the Linear Problem
        _val, back = Zygote.pullback(x -> f(x, p, nothing), y)
        s_val = size(_val)
        op = ZygotePullbackMultiplyOperator{eltype(y), typeof(back), typeof(s_val)}(back,
                                                                                    s_val)
        linear_problem = LinearProblem(op, vec(diffcache.dg_val))
        λ = solve(linear_problem, sensealg.linsolve).u
    elseif check_adjoint_mode(sensealg, Val(:jfb))
        # Jacobian Free Backpropagation
        λ = diffcache.dg_val
    else
        error("Unknown adjoint mode")
    end

    # Compute the VJP
    _, back = Zygote.pullback(p -> vec(f(y, p, nothing)), p)
    dp = back(vec(λ))[1]

    return neg(dp)
end

function DiffEqBase._concrete_solve_adjoint(prob::SteadyStateProblem, alg,
                                            sensealg::DeepEquilibriumAdjoint, u0, p,
                                            args...; save_idxs=nothing, kwargs...)
    _prob = remake(prob; u0=u0, p=p)
    sol = solve(_prob, alg, args...; kwargs...)
    _save_idxs = save_idxs === nothing ? Colon() : save_idxs

    out = save_idxs === nothing ? sol :
          DiffEqBase.sensitivity_solution(sol, sol[_save_idxs])

    function steadystatebackpass(Δ)
        # Δ = dg/dx or diffcache.dg_val
        # del g/del p = 0
        dp = adjoint_sensitivities(sol, alg; sensealg=sensealg, g=nothing, dg=Δ,
                                   save_idxs=save_idxs)
        return (NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                dp,
                NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
    end

    return out, steadystatebackpass
end

function SciMLSensitivity._adjoint_sensitivities(sol, sensealg::DeepEquilibriumAdjoint,
                                                  alg, g, dg=nothing; abstol=1e-6,
                                                  reltol=1e-3, kwargs...)
    return SciMLSensitivity.SteadyStateAdjointProblem(sol, sensealg, g, dg; kwargs...)
end

function SciMLSensitivity._adjoint_sensitivities(sol, sensealg::DeepEquilibriumAdjoint,
                                                  alg; g=nothing, dg=nothing, abstol=1e-6,
                                                  reltol=1e-3, kwargs...)
    return SciMLSensitivity.SteadyStateAdjointProblem(sol, sensealg, g, dg; kwargs...)
end
