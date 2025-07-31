@generated function split_and_reshape(x::AbstractMatrix, ::Val{idxs}, ::Val{shapes}) where {
        idxs, shapes}
    dims = [reshape((idxs[i] + 1):idxs[i + 1], shapes[i]...) for i in 1:(length(idxs) - 1)]
    varnames = map(_ -> gensym("x_view"), dims)
    calls = [:($(varnames[i]) = x[$(dims[i]), :]) for i in eachindex(dims)]
    return quote
        $(calls...)
        return tuple($(varnames...))
    end
end
split_and_reshape(x::AbstractMatrix, ::Nothing, ::Nothing) = x
split_and_reshape(x::AbstractArray, ::Nothing, ::Nothing) = x

function split_and_reshape(y::AbstractMatrix, x)
    szs = [prod(size(xᵢ)[1:(end - 1)]) for xᵢ in x]
    counters = vcat(0, cumsum(szs)[1:(end - 1)])
    # Make the data contiguous
    return map((sz, c, xᵢ) -> copy(reshape(view(y, (c + 1):(c + sz), :), size(xᵢ))), szs, counters, x)
end

flatten(x::AbstractVector) = reshape(x, length(x), 1)
flatten(x::AbstractMatrix) = x
flatten(x::AbstractArray) = reshape(x, :, size(x, ndims(x)))

flatten_vcat(x) = mapreduce(flatten, vcat, x)

function CRC.rrule(::typeof(flatten_vcat), x)
    y = flatten_vcat(x)
    project_x = CRC.ProjectTo(x)
    ∇flatten_vcat = @closure ∂y -> begin
        ∂y isa CRC.NoTangent && return (CRC.NoTangent(), CRC.NoTangent())
        return CRC.NoTangent(), project_x(split_and_reshape(∂y, x))
    end
    return y, ∇flatten_vcat
end

check_unrolled_mode(::Val{d}) where {d} = Val(d ≥ 1)
check_unrolled_mode(st::NamedTuple) = check_unrolled_mode(st.fixed_depth)

get_unrolled_depth(::Val{d}) where {d} = d
get_unrolled_depth(st::NamedTuple) = get_unrolled_depth(st.fixed_depth)

CRC.@non_differentiable check_unrolled_mode(::Any)
CRC.@non_differentiable get_unrolled_depth(::Any)

get_nfe(sol::ODESolution) = get_nfe(sol.stats)
function get_nfe(sol::NonlinearSolution)
    return ifelse(sol.stats === nothing,
        ifelse(sol.original === nothing, -1, get_nfe(sol.original)), get_nfe(sol.stats))
end
get_nfe(stats) = -1
get_nfe(stats::Union{SciMLBase.NLStats, SciMLBase.DEStats}) = stats.nf

problem_type_to_symbol(::Type{<:SteadyStateProblem{false}}) = static(:SteadyState)
problem_type_to_symbol(::Type{<:ODEProblem{false}}) = static(:ODE)

normalize_alg(deq::DEQ) = normalize_alg(deq.kind, deq.solver)
normalize_alg(_, alg) = alg
normalize_alg(::StaticSymbol{:SteadyState}, alg::AbstractODEAlgorithm) = DynamicSS(alg)
function normalize_alg(::StaticSymbol{:SteadyState}, alg::AbstractNonlinearAlgorithm)
    return SSRootfind(alg)
end
normalize_alg(::StaticSymbol{:ODE}, alg::AbstractODEAlgorithm) = alg

get_steady_state(sol::ODESolution) = last(sol.u)
get_steady_state(sol::NonlinearSolution) = sol.u
get_steady_state(sol::AbstractArray) = sol

function construct_prob(::StaticSymbol{:SteadyState}, f, u₀, p)
    return SteadyStateProblem{false}(f, u₀, p)
end
construct_prob(::StaticSymbol{:ODE}, f, u₀, p) = ODEProblem{false}(f, u₀, (0.0f0, 1.0f0), p)

function zeros_init(::Val{scales}, x::AbstractArray) where {scales}
    u₀ = similar(x, sum(prod, scales), size(x, ndims(x)))
    fill!(u₀, false)
    return u₀
end
zeros_init(::Nothing, x::AbstractArray) = zero(x)

CRC.@non_differentiable zeros_init(::Any, ::Any)

## Don't rely on SciMLSensitivity's choice
function default_sensealg(::SteadyStateProblem)
    # Ideally we should use GMRES here, but it is not very robust
    return SteadyStateAdjoint(;
        linsolve=nothing, linsolve_kwargs=(; maxiters=10, abstol=1e-3, reltol=1e-3),
        autojacvec=ZygoteVJP())
end
default_sensealg(::ODEProblem) = GaussAdjoint(; autojacvec=ZygoteVJP())

function randn_like(rng::AbstractRNG, x::AbstractArray)
    y = similar(x)::typeof(x)
    randn!(rng, y)
    return y
end

CRC.@non_differentiable randn_like(::Any...)

tupleify(x) = @closure(u->(u, x))

# Jacobian Stabilization
function estimate_jacobian_trace(::AutoFiniteDiff, model::StatefulLuxLayer, z, x, rng)
    __f = @closure u -> model((u, x))
    res = zero(eltype(x))
    ϵ = cbrt(eps(typeof(res)))
    ϵ⁻¹ = inv(ϵ)
    f₀ = __f(z)
    v = randn_like(rng, x)

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

function estimate_jacobian_trace(ad::AutoZygote, model::StatefulLuxLayer, z, x, rng)
    v = randn_like(rng, x)
    smodel = model ∘ tupleify(x)
    vjp = Lux.vector_jacobian_product(smodel, ad, z, v)
    return sum(reshape(vjp, 1, :, size(vjp, ndims(vjp))) ⊠
               reshape(v, :, 1, size(v, ndims(v))))
end

function estimate_jacobian_trace(ad::AutoForwardDiff, model::StatefulLuxLayer, z, x, rng)
    v = randn_like(rng, x)
    smodel = model ∘ tupleify(x)
    jvp = Lux.jacobian_vector_product(smodel, ad, z, v)
    return sum(reshape(v, 1, :, size(v, ndims(v))) ⊠
               reshape(jvp, :, 1, size(jvp, ndims(jvp))))
end

estimate_jacobian_trace(::Nothing, model, z, x, rng) = zero(eltype(x))
