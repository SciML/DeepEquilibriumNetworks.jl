struct LinSolveKrylovJL{S,A,T,K}
    solver::S
    args::A
    atol::T
    rtol::T
    kwargs::K
end

LinSolveKrylovJL(
    solver = LinearSolve.gmres,
    args...;
    atol::T = √eps(Float32),
    rtol::T = √eps(Float32),
    kwargs...,
) where {T} = LinSolveKrylovJL(solver, args, atol, rtol, kwargs)

function (l::LinSolveKrylovJL)(
    x,
    A,
    b::AbstractVector{T},
    matrix_updated = false,
) where {T}
    prob = LinearProblem(A, b)
    solver = KrylovJL(l.solver, A, b, nothing, nothing)
    x .=
        solve(
            prob,
            solver,
            l.args...;
            atol = T(l.atol),
            rtol = T(l.rtol),
            l.kwargs...,
        ).u
    return x
end

(l::LinSolveKrylovJL)(::Type{Val{:init}}, f, u0_prototype::AbstractVector) = l
