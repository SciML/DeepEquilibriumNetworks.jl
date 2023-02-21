import LinearAlgebra
import LinearSolve
import SciMLSensitivity
import Static

# General DEQ Utils
"""
    DeepEquilibriumAdjoint(reltol, abstol, maxiters; autojacvec=ZygoteVJP(),
                           linsolve=KrylovJL_GMRES(; rtol=reltol, atol=abstol,
                                                   itmax=maxiters),
                           mode=:vanilla)

Creates DeepEquilibriumAdjoint ([johnson2012notes](@cite)) with sensible defaults.

## Arguments

  - `reltol`: Relative tolerance.
  - `abstol`: Absolute tolerance.
  - `maxiters`: Maximum number of iterations.
  - `autojacvec`: Which backend to use for VJP.
  - `linsolve`: Linear Solver from
    [LinearSolve.jl](https://docs.sciml.ai/LinearSolve/stable/).
  - `mode`: Adjoint mode. Currently, only `:vanilla` & `:jfb` are supported.
"""
struct DeepEquilibriumAdjoint{CS, AD, FDT, M, VJP, LS} <:
       SciMLSensitivity.AbstractAdjointSensitivityAlgorithm{CS, AD, FDT}
  autojacvec::VJP
  linsolve::LS
end

@inline function _check_adjoint_mode(::DeepEquilibriumAdjoint{CS, AD, FDT, M},
                                     ::Val{M}) where {CS, AD, FDT, M}
  return true
end
@inline _check_adjoint_mode(::DeepEquilibriumAdjoint, ::Val) = false

function DeepEquilibriumAdjoint(reltol, abstol, maxiters;
                                autojacvec=SciMLSensitivity.ZygoteVJP(),
                                linsolve=LinearSolve.KrylovJL_GMRES(; rtol=reltol,
                                                                    atol=abstol,
                                                                    itmax=maxiters),
                                autodiff=true, chunk_size=0, diff_type=Val{:central},
                                mode::Symbol=:vanilla)
  @assert mode in (:vanilla, :jfb)
  return DeepEquilibriumAdjoint{chunk_size, autodiff, diff_type, mode, typeof(autojacvec),
                                typeof(linsolve)}(autojacvec, linsolve)
end

# For MultiScale DEQs
"""
    split_and_reshape(x::AbstractMatrix, ::Sidxs, ::Sshapes)

Splits up the AbstractMatrix into chunks and reshapes them.

## Arguments

  - `x`: Matrix to be split up.
  - `Sidxs`: Indices to partition the array at. (must be a `Static.static` type).
  - `Sshapes`: Shapes to reshape the split the arrays. (must be a `Static.static` type).

## Example

```@example
import DeepEquilibriumNetworks as DEQs
import Static

x1 = ones(Float32, 4, 4)
x2 = fill!(zeros(Float32, 2, 4), 0.5f0)
x3 = zeros(Float32, 1, 4)

x = vcat(x1, x2, x3)
split_idxs = Static.static(cumsum((0, size(x1, 1), size(x2, 1), size(x3, 1))))
shapes = Static.static((size(x1, 1), size(x2, 1), size(x3, 1)))

DEQs.split_and_reshape(x, split_idxs, shapes)
```
"""
@generated function split_and_reshape(x::AbstractMatrix, ::T, ::S) where {T, S}
  idxs, shapes = Static.known(T), Static.known(S)
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
"""
    init_identity_matrix(x::AbstractArray, scale::T=1)

Create an identity matrix of shape `[length(x), length(x)]` and placed on the same device
as `x`, and scale the matrix by `scale`.
"""
@inline function init_identity_matrix(x::AbstractArray{T}, scale::T=T(1)) where {T}
  x_ = vec(x)
  return _init_identity_matrix!(x_ .* x_', scale)
end

@inline function _init_identity_matrix!(x::AbstractMatrix{T}, scale::T=T(1)) where {T}
  x .= zero(T)
  view(x, LinearAlgebra.diagind(x)) .= scale .* true
  return x
end
