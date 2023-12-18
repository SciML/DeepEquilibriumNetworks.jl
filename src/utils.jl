@generated function __split_and_reshape(x::AbstractMatrix, ::Val{idxs},
        ::Val{shapes}) where {idxs, shapes}
    dims = [reshape((idxs[i] + 1):idxs[i + 1], shapes[i]...) for i in 1:(length(idxs) - 1)]
    varnames = map(_ -> gensym("x_view"), dims)
    calls = [:($(varnames[i]) = view(x, $(dims[i]), :)) for i in 1:length(dims)]
    return quote
        $(calls...)
        return tuple($(varnames...))
    end
end
__split_and_reshape(x::AbstractMatrix, ::Nothing, ::Nothing) = x
__split_and_reshape(x::AbstractArray, ::Nothing, ::Nothing) = x

function __split_and_reshape(y::AbstractMatrix, x)
    szs = [prod(size(xᵢ)[1:(end - 1)]) for xᵢ in x]
    counters = vcat(0, cumsum(szs)[1:(end - 1)])
    return map((sz, c, xᵢ) -> reshape(view(y, (c + 1):(c + sz), :), size(xᵢ)),
        szs, counters, x)
end

@inline __flatten(x::AbstractVector) = reshape(x, length(x), 1)
@inline __flatten(x::AbstractMatrix) = x
@inline __flatten(x::AbstractArray) = reshape(x, :, size(x, ndims(x)))

@inline __flatten_vcat(x) = mapreduce(__flatten, vcat, x)

function CRC.rrule(::typeof(__flatten_vcat), x)
    y = __flatten_vcat(x)
    project_x = CRC.ProjectTo(x)
    function ∇__flatten_vcat(∂y)
        ∂y isa CRC.NoTangent && return (CRC.NoTangent(), CRC.NoTangent())
        return CRC.NoTangent(), project_x(__split_and_reshape(∂y, x))
    end
    return y, ∇__flatten_vcat
end

@inline __check_unrolled_mode(::Val{d}) where {d} = Val(d ≥ 1)
@inline __check_unrolled_mode(st::NamedTuple) = __check_unrolled_mode(st.fixed_depth)

@inline __get_unrolled_depth(::Val{d}) where {d} = d
@inline __get_unrolled_depth(st::NamedTuple) = __get_unrolled_depth(st.fixed_depth)

CRC.@non_differentiable __check_unrolled_mode(::Any)
CRC.@non_differentiable __get_unrolled_depth(::Any)

@inline @generated function __getproperty(obj, ::Val{field}) where {field}
    hasfield(obj, field) && return :(obj.$field)
    return :(nothing)
end

@inline __get_nfe(sol::ODESolution) = __get_nfe(sol.stats)
@inline function __get_nfe(sol::NonlinearSolution)
    return ifelse(sol.stats === nothing,
        ifelse(sol.original === nothing, -1, __get_nfe(sol.original)), __get_nfe(sol.stats))
end
@inline __get_nfe(stats) = -1
@inline __get_nfe(stats::Union{SciMLBase.NLStats, SciMLBase.DEStats}) = stats.nf

@inline __normalize_alg(deq::DEQ{pType}) where {pType} = __normalize_alg(pType, deq.solver)
@inline __normalize_alg(::Type{<:SteadyStateProblem}, alg) = alg
@inline __normalize_alg(::Type{<:SteadyStateProblem}, alg::AbstractODEAlgorithm) = DynamicSS(alg)
@inline __normalize_alg(::Type{<:SteadyStateProblem}, alg::AbstractNonlinearAlgorithm) = SSRootfind(alg)
@inline __normalize_alg(::Type{<:ODEProblem}, alg::AbstractODEAlgorithm) = alg

@inline __get_steady_state(sol::ODESolution) = last(sol.u)
@inline __get_steady_state(sol::NonlinearSolution) = sol.u
@inline __get_steady_state(sol::AbstractArray) = sol

@inline function __construct_prob(::Type{<:SteadyStateProblem{false}}, f, u₀, p)
    return SteadyStateProblem{false}(f, u₀, p)
end
@inline function __construct_prob(::Type{<:ODEProblem{false}}, f, u₀, p)
    return ODEProblem{false}(f, u₀, (0.0f0, 1.0f0), p)
end

@inline function __zeros_init(::Val{scales}, x::AbstractArray) where {scales}
    u₀ = similar(x, sum(prod, scales), size(x, ndims(x)))
    fill!(u₀, false)
    return u₀
end
@inline __zeros_init(::Nothing, x::AbstractArray) = zero(x)

CRC.@non_differentiable __zeros_init(::Any, ::Any)

## Don't rely on SciMLSensitivity's choice
@inline __default_sensealg(prob) = nothing

@inline function __gaussian_like(rng::AbstractRNG, x)
    y = similar(x)
    randn!(rng, y)
    return y
end

CRC.@non_differentiable __gaussian_like(::Any...)

# Jacobian Stabilization
function __estimate_jacobian_trace(::AutoFiniteDiff, model, ps, z, x, rng)
    __f = u -> first(model((u, x), ps))
    res = zero(eltype(x))
    ϵ = cbrt(eps(typeof(res)))
    ϵ⁻¹ = inv(ϵ)
    f₀ = __f(z)
    v = __gaussian_like(rng, x)

    for idx in eachindex(z)
        _z = z[idx]
        CRC.ignore_derivatives() do
            z[idx] = z[idx] + ϵ
        end
        res = res + abs2(sum((__f(z) .- f₀) .* ϵ⁻¹ .* v)) / length(v)
        CRC.ignore_derivatives() do
            z[idx] = _z
        end
    end

    return res
end

__estimate_jacobian_trace(::Nothing, model, ps, z, x, rng) = zero(eltype(x))
