# General DEQ Utils
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
    return (rng::AbstractRNG, dims...) -> randn(rng, Float32, dims...) .* σ² .+ μ
end

# For MultiScale DEQs
function split_and_reshape(x::AbstractMatrix, idxs::Tuple, shapes::Tuple)
    return Tuple(
        @view(x[reshape((idxs[i] + 1):idxs[i + 1], shapes[i]...), :]) for i in 1:(length(idxs) - 1)
    )
end

## Some dispatches for CuArrays are not defined for subarrays
# function split_and_reshape(x::AbstractMatrix, idxs::Tuple, shapes::Tuple)
#     return Tuple(
#         x[reshape((idxs[i] + 1):idxs[i + 1], shapes[i]...), :] for i in 1:(length(idxs) - 1)
#     )
# end

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
