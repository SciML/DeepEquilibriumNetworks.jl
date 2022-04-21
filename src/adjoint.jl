neg(x::Any) = hasmethod(-, (typeof(x),)) ? -x : x
neg(nt::NamedTuple) = fmap(neg, nt)

@noinline function DiffEqSensitivity.SteadyStateAdjointProblem(
    sol::EquilibriumSolution, sensealg::DiffEqSensitivity.SteadyStateAdjoint, g::Nothing, dg; save_idxs=nothing
)
    @unpack f, p, u0 = sol.prob

    diffcache, y = DiffEqSensitivity.adjointdiffcache(g, sensealg, false, sol, dg, f; quad=false, needs_jac=false)

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

    # Solve the Linear Problem
    _val, back = Zygote.pullback(x -> f(x, p, nothing), y)
    s_val = size(_val)
    op = ZygotePullbackMultiplyOperator{eltype(y),typeof(back),typeof(s_val)}(back, s_val)
    linear_problem = LinearProblem(op, vec(diffcache.dg_val))
    ## Automatically choose the best algorithm
    λ = solve(linear_problem, sensealg.linsolve).u

    # Compute the VJP
    _, back = Zygote.pullback(p -> vec(f(y, p, nothing)), p)
    dp = back(vec(λ))[1]

    return neg(dp)
end
