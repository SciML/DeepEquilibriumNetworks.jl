# Type Stable GroupNorm
mutable struct NormAttributes
    affine::Bool
    track_stats::Bool
    active::Union{Bool,Nothing}
end

Flux.hasaffine(attr::NormAttributes) = attr.affine

struct GroupNormV2{F,V,N,W}
    G::Int  # number of groups
    λ::F  # activation function
    β::V  # bias
    γ::V  # scale
    μ::W     # moving mean
    σ²::W    # moving std
    ϵ::N
    momentum::N
    chs::Int # number of channels
    attrs::NormAttributes
end

Flux.hasaffine(gn::GroupNormV2) = Flux.hasaffine(gn.attrs)
Flux._isactive(gn::GroupNormV2) = Flux._isactive(gn.attrs)

Flux.gpu(gn::GroupNormV2) = GroupNormV2(
    gn.G,
    gn.λ,
    Flux.gpu(gn.β),
    Flux.gpu(gn.γ),
    Flux.gpu(gn.μ),
    Flux.gpu(gn.σ²),
    gn.ϵ,
    gn.momentum,
    gn.chs,
    gn.attrs,
)

Flux.cpu(gn::GroupNormV2) = GroupNormV2(
    gn.G,
    gn.λ,
    Flux.cpu(gn.β),
    Flux.cpu(gn.γ),
    Flux.cpu(gn.μ),
    Flux.cpu(gn.σ²),
    gn.ϵ,
    gn.momentum,
    gn.chs,
    gn.attrs,
)

Flux.@functor GroupNormV2 (β, γ)

Flux.trainable(gn::GroupNormV2) = hasaffine(gn) ? (gn.β, gn.γ) : ()

function GroupNormV2(
    chs::Int,
    G::Int,
    λ = identity;
    initβ = zeros32,
    initγ = ones32,
    affine = true,
    track_stats = false,
    ϵ = 1f-5,
    momentum = 0.1f0,
)
    @assert chs % G == 0 "The number of groups ($(G)) must divide the number of channels ($chs)"

    β = affine ? initβ(chs) : nothing
    γ = affine ? initγ(chs) : nothing
    μ = track_stats ? zeros32(G) : nothing
    σ² = track_stats ? ones32(G) : nothing

    return GroupNormV2(
        G,
        λ,
        β,
        γ,
        μ,
        σ²,
        ϵ,
        momentum,
        chs,
        NormAttributes(affine, track_stats, nothing),
    )
end

function get_stats(::Val{true}, ::Val{false}, l::GroupNormV2, x::AbstractArray{T,N}, reduce_dims) where {T,N}
    # testmode with tracked stats
    stats_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
    return reshape(l.μ, stats_shape), reshape(l.σ², stats_shape)
end

function get_stats(::Val{false}, active, ::GroupNormV2, x, reduce_dims)
    # trainmode or testmode without tracked stats
    μ = mean(x; dims = reduce_dims)
    diff = x .- μ
    return μ, mean(abs2, diff; dims = reduce_dims)
end

function get_stats(::Val{true}, active::Val{true}, l::GroupNormV2, x::AbstractArray{T,N}, reduce_dims) where {T,N}
    # trainmode with tracked stats
    μ, σ² = get_stats(Val(false), active, l, x, reduce_dims)
    Zygote.ignore() do
        mtm = l.momentum
        m = prod(size(x)[reduce_dims])  # needed for computing corrected var
        μnew = vec(N ∈ reduce_dims ? μ : mean(μ, dims = N))
        σ²new = vec(N ∈ reduce_dims ? σ² : mean(σ², dims = N))
        l.μ .= (1 - mtm) .* l.μ .+ mtm .* μnew
        l.σ² .=
            (1 - mtm) .* l.σ² .+
            mtm .* (m / (m - one(eltype(l.σ²)))) .* σ²new
    end
    return μ, σ²
end

function group_norm_forward(
    l,
    x::AbstractArray{T,N},
    reduce_dims,
    affine_shape,
) where {T,N}
    μ, σ² = get_stats(Val(l.attrs.track_stats), Val(_isactive(l)), l, x, reduce_dims)
    if hasaffine(l)
        γ = reshape(l.γ, affine_shape)
        β = reshape(l.β, affine_shape)
        return l.λ.(norm_forward(μ, σ², x, γ, β, l.ϵ))
    else
        return l.λ.(norm_forward(μ, σ², x, l.ϵ))
    end
end

norm_forward(μ, σ², x, ϵ) = (x .- μ) ./ sqrt.(σ² .+ ϵ)

norm_forward(μ, σ², x, γ, β, ϵ) = γ .* (x .- μ) ./ sqrt.(σ² .+ ϵ) .+ β

Zygote.@adjoint function norm_forward(μ, σ², x, γ, β, ϵ)
    N = ndims(x)

    σ²ϵ = σ² .+ ϵ
    inv_deno = 1 ./ sqrt.(σ²ϵ)
    res_1 = (x .- μ) .* inv_deno
    res_2 = γ .* res_1
    res = res_2 .+ β

    function norm_backward(Δ)
        reduce_dims_affine = filter(i -> isone(size(β, i)), 1:N)
        reduce_dims_stats = filter(i -> isone(size(σ², i)), 1:N)

        Δx = inv_deno .* Δ
        Δμ = -sum(Δx; dims = reduce_dims_stats)
        Δσ² = sum(-eltype(x)(0.5) .* res_2 .* Δ ./ σ²ϵ; dims = reduce_dims_stats)
        Δγ = sum(res_1 .* Δ; dims = reduce_dims_affine)
        Δβ = sum(Δ; dims = reduce_dims_affine)

        return (Δμ, Δσ², Δx, Δγ, Δβ, nothing)
    end

    return res, norm_backward
end

Zygote.@adjoint function norm_forward(μ, σ², x, ϵ)
    N = ndims(x)

    σ²ϵ = σ² .+ ϵ
    inv_deno = 1 ./ sqrt.(σ²ϵ)
    res = (x .- μ) .* inv_deno

    function norm_backward(Δ)
        reduce_dims_stats = filter(i -> isone(size(σ², i)), 1:N)

        Δx = inv_deno .* Δ
        Δμ = -sum(Δx; dims = reduce_dims_stats)
        Δσ² = sum(-eltype(x)(0.5) .* res_2 .* Δ ./ σ²ϵ; dims = reduce_dims_stats)

        return (Δμ, Δσ², Δx, nothing)
    end

    return res, norm_backward
end

function (gn::GroupNormV2)(x::AbstractArray{T,N}) where {T,N}
    # Not doing assertion checks
    # @assert N > 2
    # @assert size(x, N - 1) == gn.chs
    sz = size(x)
    x_2 = reshape(x, sz[1:N-2]..., sz[N-1] ÷ gn.G, gn.G, sz[N])
    N_ = ndims(x_2)
    reduce_dims = 1:N_-2
    affine_shape = ntuple(i -> i ∈ (N_ - 1, N_ - 2) ? size(x_2, i) : 1, N_)
    x_3 = group_norm_forward(gn, x_2, reduce_dims, affine_shape)
    return reshape(x_3, sz)
end

testmode!(m::GroupNormV2, mode = true) =
    (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)

function Base.show(io::IO, l::GroupNormV2)
    print(io, "GroupNormV2($(l.chs), $(l.G)")
    l.λ == identity || print(io, ", ", l.λ)
    hasaffine(l) || print(io, ", affine=false")
    print(io, ")")
end
