@generated function _evaluate_unrolled_model(::AbstractDEQs, model, z_star, x, ps, st,
        ::Val{d}) where {d}
    calls = [:((z_star, st) = model((z_star, x), ps, st)) for _ in 1:d]
    push!(calls, :(return z_star, st))
    return Expr(:block, calls...)
end

function build_solution(deq::AbstractDEQs, z_star, z, x, ps, st, nfe, jac_loss)
    residual = CRC.ignore_derivatives(z_star .-
                                      deq.model((z_star, x), ps.model, st.model)[1])
    return DeepEquilibriumSolution(z_star, z, residual, jac_loss, nfe)
end

@inline _postprocess_output(_, z_star) = z_star

@inline function _construct_problem(::AbstractDEQs, dudt, z, ps, x)
    return SteadyStateProblem(ODEFunction{false}(dudt), z, (; ps=ps.model, x))
end

@inline _fix_solution_output(_, x) = x

function (deq::AbstractDEQs)(x::AbstractArray{T}, ps, st::NamedTuple, ::Val{true}) where {T}
    # Pretraining without Fixed Point Solving
    z, st = _get_initial_condition(deq, x, ps, st)
    depth = _get_unrolled_depth(st)

    z_star, st_ = _evaluate_unrolled_model(deq, deq.model, z, x, ps.model, st.model,
        st.fixed_depth)

    st__ = merge(st,
        (; model=st_, solution=build_solution(deq, z_star, z, x, ps, st, depth, T(0))))

    return _postprocess_output(deq, z_star), st__
end

function (deq::AbstractDEQs)(x::AbstractArray, ps, st::NamedTuple, ::Val{false})
    T = eltype(x)
    z, st = _get_initial_condition(deq, x, ps, st)

    model = Lux.Experimental.StatefulLuxLayer(deq.model, nothing, st.model)
    nfe::Int = 0

    function dudt(u, p, t)
        nfe += 1
        return model((u, p.x), p.ps) .- u
    end

    prob = _construct_problem(deq, dudt, z, ps, x)
    sol = solve(prob, deq.solver; deq.sensealg, deq.kwargs...)
    _z_star = sol.u
    # Handle Neural ODEs
    z_star = _z_star isa Vector{<:AbstractArray} ? last(_z_star) : _z_star

    if _jacobian_regularization(deq)
        rng = Lux.replicate(st.rng)
        jac_loss = estimate_jacobian_trace(Val(:finite_diff), deq.model, ps.model, model.st,
            z_star, x, rng)
    else
        rng = st.rng
        jac_loss = T(0)
    end

    st_ = merge(st,
        (; model=model.st, rng,
            solution=build_solution(deq, z_star, z, x, ps, st, nfe, jac_loss)))

    return _postprocess_output(deq, z_star), st_
end
