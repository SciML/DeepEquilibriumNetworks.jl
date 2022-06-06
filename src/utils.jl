# General DEQ Utils
"""
    DeepEquilibriumAdjoint(reltol, abstol, maxiters; autojacvec=ZygoteVJP(),
                           linsolve=KrylovJL_GMRES(; rtol=reltol, atol=abstol, itmax=maxiters),
                           mode=:vanilla)

Creates DeepEquilibriumAdjoint ([johnson2012notes](@cite)) with sensible defaults.

## Arguments

* `reltol`: Relative tolerance.
* `abstol`: Absolute tolerance.
* `maxiters`: Maximum number of iterations.
* `autojacvec`: Which backend to use for VJP.
* `linsolve`: Linear Solver from [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl).
* `mode`: Adjoint mode. Currently only `:vanilla` & `:jfb` are supported.
"""
struct DeepEquilibriumAdjoint{CS, AD, FDT, M, VJP, LS} <:
       AbstractAdjointSensitivityAlgorithm{CS, AD, FDT}
    autojacvec::VJP
    linsolve::LS
end

@inline function check_adjoint_mode(::DeepEquilibriumAdjoint{CS, AD, FDT, M},
                                    ::Val{M}) where {CS, AD, FDT, M}
    true
end
@inline check_adjoint_mode(::DeepEquilibriumAdjoint, ::Val) = false

Base.@pure function DeepEquilibriumAdjoint(reltol,
                                           abstol,
                                           maxiters;
                                           autojacvec=ZygoteVJP(),
                                           linsolve=KrylovJL_GMRES(; rtol=reltol,
                                                                   atol=abstol,
                                                                   itmax=maxiters),
                                           autodiff=true,
                                           chunk_size=0,
                                           diff_type=Val{:central},
                                           mode::Symbol=:vanilla)
    return DeepEquilibriumAdjoint{
                                  chunk_size, autodiff, diff_type, mode, typeof(autojacvec),
                                  typeof(linsolve)
                                  }(autojacvec, linsolve)
end

# Initialization
"""
    NormalInitializer(μ = 0.0f0, σ² = 0.01f0)

Initializes the weights of the network with a normal distribution. For DEQs the training is stable
if we use this as the Initialization
"""
function NormalInitializer(μ=0.0f0, σ²=0.01f0)
    return (rng::AbstractRNG, dims...) -> randn(rng, Float32, dims...) .* σ² .+ μ
end

# For MultiScale DEQs
@generated function split_and_reshape(x::AbstractMatrix, ::T, ::S) where {T, S}
    idxs, shapes = known(T), known(S)
    dims = [reshape((idxs[i] + 1):idxs[i + 1], shapes[i]...) for i in 1:(length(idxs) - 1)]
    varnames = [gensym("x_view") for _ in dims]
    calls = []
    for (i, dim) in enumerate(dims)
        push!(calls, :($(varnames[i]) = view(x, $dim, :)))
    end
    push!(calls, :(return tuple($(Tuple(varnames)...))))
    return Expr(:block, calls...)
end

# General Utils
@inline function _init_identity_matrix(x::AbstractArray{T}, scale::T=T(1)) where {T}
    x_ = vec(x)
    return _init_identity_matrix!(x_ .* x_', scale)
end

@inline function _init_identity_matrix!(x::AbstractMatrix{T}, scale::T=T(1)) where {T}
    x .= zero(T)
    view(x, diagind(x)) .= scale .* true
    return x
end

@inline _norm(x; dims=Colon()) = sqrt.(sum(abs2, x; dims=dims))

# Compute norm over all dimensions except `except_dim`
@inline function _norm(x::AbstractArray{T, N}, except_dim) where {T, N}
    _norm(x; dims=filter(i -> i != except_dim, 1:N))
end
