# General DEQ Utils
mutable struct DEQTrainingStats
    nfe::Int
end

"""
    get_and_clear_nfe!(model::AbstractDeepEquilibriumNetwork)

Return the number of function evaluations (NFE) and clear the counter.
"""
function get_and_clear_nfe!(model::AbstractDeepEquilibriumNetwork)
    nfe = model.stats.nfe
    model.stats.nfe = 0
    return nfe
end

"""
    SteadyStateAdjoint(reltol, abstol, maxiters; autojacvec=ZygoteVJP(),
                       linsolve=KrylovJL_GMRES(; rtol=reltol, atol=abstol, itmax=maxiters))

Creates SteadyStateAdjoint ([johnson2012notes](@cite)) with sensible defaults.

## Arguments

* `reltol`: Relative tolerance.
* `abstol`: Absolute tolerance.
* `maxiters`: Maximum number of iterations.
* `autojacvec`: Which backend to use for VJP.
* `linsolve`: Linear Solver from [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl).
"""
function DiffEqSensitivity.SteadyStateAdjoint(reltol, abstol, maxiters; autojacvec=ZygoteVJP(),
                                              linsolve=KrylovJL_GMRES(; rtol=reltol, atol=abstol, itmax=maxiters))
    return SteadyStateAdjoint(; autodiff=true, autojacvec=autojacvec, linsolve=linsolve)
end

# Initialization
"""
    NormalInitializer(μ = 0.0f0, σ² = 0.01f0)

Initializes the weights of the network with a normal distribution. For DEQs the training is stable
if we use this as the Initialization
"""
function NormalInitializer(μ = 0.0f0, σ² = 0.01f0)
    return (dims...) -> randn(dims...) .* σ² .+ μ
end

# Wrapper for Discrete DEQs
"""
    DiscreteDEQSolver(solver=LimitedMemoryBroydenSolver; abstol=1e-8, reltol=1e-8, kwargs...)

Solver for Discrete DEQ Problem ([baideep2019](@cite)). A wrapper around `SSrootfind` to mimic the [`ContinuousDEQSolver`](@ref) API. 

## Arguments

* `solver`: NonLinear Solver for the DEQ problem. (Default: [`LimitedMemoryBroydenSolver`](@ref))
* `abstol`: Absolute tolerance for termination. (Default: `1e-8`)
* `reltol`: Relative tolerance for termination. (Default: `1e-8`)
* `kwargs`: Additional keyword arguments passed to the solver.

!!! note
    There is no `mode` kwarg for [`DiscreteDEQSolver`](@ref). Instead solvers directly define their own termination condition.
    For [`BroydenSolver`](@ref) and [`LimitedMemoryBroydenSolver`](@ref), the termination conditions are `:abs_norm` &
    `:rel_deq_default` respectively.

See also: [`ContinuousDEQSolver`](@ref)
"""
function DiscreteDEQSolver(solver=LimitedMemoryBroydenSolver; abstol=1e-8, reltol=1e-8, kwargs...)
    solver = solver(; kwargs..., reltol=reltol, abstol=abstol)
    return SSRootfind(; nlsolve=(f, u0, abstol) -> solver(f, u0))
end

# For MultiScale DEQs
function split_array_by_indices(x::AbstractVector, idxs)
    return collect((x[(i + 1):j] for (i, j) in zip(idxs[1:(end - 1)], idxs[2:end])))
end

function split_array_by_indices(x::AbstractMatrix, idxs)
    return collect((x[(i + 1):j, :] for (i, j) in zip(idxs[1:(end - 1)], idxs[2:end])))
end

Zygote.@adjoint function split_array_by_indices(x, idxs)
    res = split_array_by_indices(x, idxs)
    function split_array_by_indices_sensitivity(Δ)
        is_nothings = Δ .=== nothing
        if any(is_nothings)
            Δ[is_nothings] .= zero.(res[is_nothings])
        end
        return (vcat(Δ...), nothing)
    end
    return res, split_array_by_indices_sensitivity
end

# Zygote Fix
function Zygote.accum(x::NTuple{N,T}, y::AbstractVector{T}) where {N,T<:AbstractArray}
    return Zygote.accum.(x, y)
end

function Zygote.accum(x::AbstractVector{T}, y::NTuple{N,T}) where {N,T<:AbstractArray}
    return Zygote.accum.(x, y)
end

function Zygote.accum(x::AbstractVector{T}, y::NTuple{N,Nothing}) where {N,T<:AbstractArray}
    return Zygote.accum.(x, y)
end

# General Utils
@inline function _init_identity_matrix(x::AbstractArray{T}, scale::T=T(1)) where {T}
    x_ = vec(x)
    return _init_identity_matrix!(x_ .* x_', scale)
end

@inline function _init_identity_matrix!(x::AbstractMatrix{T}, scale::T=T(1)) where {T}
    x .= zero(T)
    idxs = diagind(x)
    @. @view(x[idxs]) = scale * true
    return x
end

@inline function _norm(x; dims=Colon())
    return sqrt.(sum(abs2, x; dims=dims))
end

# Compute norm over all dimensions except `except_dim`
@inline function _norm(x::AbstractArray{T,N}, except_dim) where {T,N}
    dims = filter(i -> i != except_dim, 1:N)
    return _norm(x; dims=dims)
end

flatten_merge(x, y) = (x..., y...)
flatten_merge(x::T, y::T) where {T<:AbstractArray} = (x, y)
flatten_merge(x::NTuple{N,T}, y::T) where {N,T<:AbstractArray} = (x..., y)
flatten_merge(x::T, y::NTuple{N,T}) where {N,T<:AbstractArray} = (x, y...)
flatten_merge(x::NTuple{N,T}, y) where {N,T<:AbstractArray} = (x, y...)
flatten_merge(x, y::NTuple{N,T}) where {N,T<:AbstractArray} = (x..., y)
function flatten_merge(x::NTuple{N,T}, y::NTuple{N,T}) where {N,T<:AbstractArray}
    return (x, y)
end
