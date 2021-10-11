struct LinSolveKrylovJL{S,A,K}
    solver::S
    args::A
    kwargs::K
end

LinSolveKrylovJL(solver = LinearSolve.gmres, args...; kwargs...) =
    LinSolveKrylovJL(solver, args, kwargs)

function (l::LinSolveKrylovJL)(x, A, b, matrix_updated = false)
    prob = LinearProblem(A, b)
    solver = KrylovJL(l.solver, A, b, nothing, nothing)
    x .= solve(prob, solver, l.args...; l.kwargs...).u
    return x
end

(l::LinSolveKrylovJL)(::Type{Val{:init}}, f, u0_prototype) = l
