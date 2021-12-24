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

@inline function get_and_clear_nfe!(model::DataParallelFluxModel)
    return get_and_clear_nfe!(model.model)
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

# General DEQ Utils
function get_and_clear_nfe!(model::AbstractDeepEquilibriumNetwork)
    nfe = model.stats.nfe
    model.stats.nfe = 0
    return nfe
end

mutable struct DEQTrainingStats
    nfe::Int
end

function get_default_ssadjoint(reltol, abstol, maxiters)
    return SteadyStateAdjoint(; autodiff=true, autojacvec=ZygoteVJP(),
                              linsolve=KrylovJL_GMRES(; rtol=reltol, atol=abstol, itmax=maxiters))
end

function get_default_dynamicss_solver(reltol, abstol, ode_solver=Tsit5(); mode::Symbol = :rel_deq_default)
    # return DynamicSS(ode_solver; reltol=reltol, abstol=abstol)
    return DEQSolver(ode_solver; mode=mode, reltol=reltol, abstol=abstol)
end

function get_default_ssrootfind_solver(reltol, abstol, solver=LimitedMemoryBroydenSolver; kwargs...)
    _solver = solver(; kwargs..., reltol=reltol, abstol=abstol)
    return SSRootfind(; nlsolve=(f, u0, abstol) -> _solver(f, u0))
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

# Initialization
function normal_init(μ = 0.0f0, σ² = 0.01f0)
    return (dims...) -> randn(dims...) .* σ² .+ μ
end