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


struct LinearScaledJacVecOperator{T,JacVecOp,M} <: DiffEqBase.AbstractDiffEqLinearOperator{T}
    # Operator = mat - jacvecop
    jacvecop::JacVecOp
    mat::M
    function LinearScaledJacVecOperator(jacvecop, mat::AbstractMatrix)
        return new{eltype(mat), typeof(jacvecop), typeof(mat)}(jacvecop, mat)
    end
    function LinearScaledJacVecOperator(jacvecop, v::AbstractVector)
        mat = Diagonal(v)
        return new{eltype(mat), typeof(jacvecop), typeof(mat)}(jacvecop, mat)
    end
    function LinearScaledJacVecOperator(jacvecop, n::Integer)
        return LinearScaledJacVecOperator(jacvecop, ones(n))
    end
end

function LinearAlgebra.mul!(y::AbstractArray, l::LinearScaledJacVecOperator, b::AbstractVector)
    _y = similar(y)
    mul!(y, l.mat, b)
    mul!(_y, l.jacvecop, b)
    y .-= _y
    return y
end

(l::LinearScaledJacVecOperator)(u, p, t) = l.mat * u - l.jacvecop(u, p, t)

function (l::LinearScaledJacVecOperator)(du, u, p, t)
    du = l.jacvecop(du, u, p, t)
    du .= mat * u .- du
    return du
end

Base.size(l::LinearScaledJacVecOperator) = Base.size(l.mat)