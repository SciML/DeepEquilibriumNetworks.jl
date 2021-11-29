# General Utils
function _init_identity_matrix(x::AbstractArray{T}, scale::T = T(1)) where {T}
    x_ = vec(x)
    return _init_identity_matrix!(x_ .* x_', scale)
end

function _init_identity_matrix!(x::AbstractMatrix{T}, scale::T = T(1)) where {T}
    x .= zero(T)
    idxs = diagind(x)
    @. @view(x[idxs]) = scale * true
    return x
end

function _norm(x; dims = Colon())
    return sqrt.(sum(abs2, x; dims = dims))
end

get_and_clear_nfe!(model::DataParallelFluxModel) =
    get_and_clear_nfe!(model.model)

# Compute norm over all dimensions except `except_dim`
function _norm(x::AbstractArray{T,N}, except_dim) where {T,N}
    dims = filter(i -> i != except_dim, 1:N)
    return _norm(x; dims = dims)
end

flatten_merge(x, y) = (x..., y...)
flatten_merge(x::T, y::T) where {T<:AbstractArray} = (x, y)
flatten_merge(x::NTuple{N,T}, y::T) where {N,T<:AbstractArray} = (x..., y)
flatten_merge(x::T, y::NTuple{N,T}) where {N,T<:AbstractArray} = (x, y...)
flatten_merge(x::NTuple{N,T}, y) where {N,T<:AbstractArray} = (x, y...)
flatten_merge(x, y::NTuple{N,T}) where {N,T<:AbstractArray} = (x..., y)
flatten_merge(x::NTuple{N,T}, y::NTuple{N,T}) where {N,T<:AbstractArray} = (x, y)

Flux.gpu(p::Parallel) = Parallel(gpu(p.connection), gpu.(p.layers))
Flux.cpu(p::Parallel) = Parallel(cpu(p.connection), cpu.(p.layers))

# General DEQ Utils
Flux.trainable(deq::AbstractDeepEquilibriumNetwork) = (deq.p,)

function get_and_clear_nfe!(model::AbstractDeepEquilibriumNetwork)
    nfe = model.stats.nfe
    model.stats.nfe = 0
    return nfe
end

mutable struct DEQTrainingStats
    nfe::Int
end

function get_default_ssadjoint(reltol, abstol, maxiters)
    return SteadyStateAdjoint(
        autodiff = true,
        autojacvec = ZygoteVJP(),
        linsolve = KrylovJL_GMRES(rtol = reltol, atol = abstol,
                                  itmax = maxiters))
end

function get_default_dynamicss_solver(reltol, abstol, ode_solver = Tsit5())
    return DynamicSS(ode_solver, reltol = reltol, abstol = abstol)
end

function get_default_ssrootfind_solver(
    reltol,
    abstol,
    solver = LimitedMemoryBroydenSolver;
    kwargs...,
)
    _solver = solver(;kwargs..., reltol = reltol, abstol = abstol)
    return SSRootfind(; nlsolve = (f, u0, abstol) -> _solver(f, u0))
end

# For MultiScale DEQs
split_array_by_indices(x::AbstractVector, idxs) =
    collect((x[i + 1:j] for (i, j) ∈ zip(idxs[1:end-1], idxs[2:end])))

split_array_by_indices(x::AbstractMatrix, idxs) =
    collect((x[i + 1:j, :] for (i, j) ∈ zip(idxs[1:end-1], idxs[2:end])))


# Zygote Fix
Zygote.accum(x::NTuple{N,T}, y::AbstractVector{T}) where {N,T<:AbstractArray} =
    Zygote.accum.(x, y)

Zygote.accum(x::AbstractVector{T}, y::NTuple{N,T}) where {N,T<:AbstractArray} =
    Zygote.accum.(x, y)