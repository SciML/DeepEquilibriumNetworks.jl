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


struct BroydenCache{J,F,X}
    Jinv::J
    fx::F
    Δfx::F
    fx_old::F
    x::X
    Δx::X
    x_old::X
end

function BroydenCache(x)
    fx, Δfx, fx_old = copy(x), copy(x), copy(x)
    x, Δx, x_old = copy(x), copy(x), copy(x)
    Jinv = _init_identity_matrix(x)
    return BroydenCache(Jinv, fx, Δfx, fx_old, x, Δx, x_old)
end


function broyden(
    f!,
    x_::AbstractVector{T};
    cache::BroydenCache = BroydenCache(x_),
    max_iter::Int = 1000,
    abstol = 1e-3,
    reltol = 1e-3,
) where {T}
    @unpack Jinv, fx, Δfx, fx_old, x, Δx, x_old = cache
    if size(x) != size(x_)
        # This might happen when the last batch with insufficient batch_size
        # is passed.
        @unpack Jinv, fx, Δfx, fx_old, x, Δx, x_old = BroydenCache(x_)
    end
    x .= x_

    f!(fx, x)
    _init_identity_matrix!(Jinv)

    maybe_stuck = false
    max_resets = 3
    resets = 0

    for _ = 1:max_iter
        x_old .= x
        fx_old .= fx

        p = -Jinv * fx_old

        ρ, σ₂ = T(0.9), T(0.001)

        x .= x_old .+ p
        f!(fx, x)

        if norm(fx, 2) ≤ ρ * norm(fx_old, 2) - σ₂ * norm(p, 2)^2
            α = T(1)
        else
            α = _approximate_norm_descent(f!, fx, x, p)
            x .= x_old .+ α * p
            f!(fx, x)
        end

        Δx .= x .- x_old
        Δfx .= fx .- fx_old

        maybe_stuck = all(abs.(Δx) .<= eps(T)) || all(abs.(Δfx) .<= eps(T))
        if maybe_stuck
            Jinv = _init_identity_matrix(x)
            resets += 1
            maybe_stuck = (resets ≤ max_resets) && maybe_stuck
        else
            ΔxJinv = Δx' * Jinv
            Jinv .+= ((Δx .- Jinv * Δfx) ./ (ΔxJinv * Δfx)) * ΔxJinv
        end

        maybe_stuck = false

        # Convergence Check
        Δfx_norm = norm(Δfx, 2)
        (Δfx_norm ≤ abstol || Δfx_norm ./ min(norm(fx), norm(Δfx)) ≤ reltol) && return x
    end

    return x
end


# https://doi.org/10.1080/10556780008805782
# FIXME: We are dropping some robustness tests for now.
function _approximate_norm_descent(
    f!,
    fx::AbstractArray{T,N},
    x::AbstractArray{T,N},
    p;
    λ₀ = T(1),
    β = T(0.5),
    σ₁ = T(0.001),
    η = T(0.1),
    max_iter = 50,
) where {T,N}
    λ₂, λ₁ = λ₀, λ₀

    f!(fx, x)
    fx_norm = norm(fx, 2)

    # TODO: Test NaN/Finite
    # f!(fx, x .- λ₂ .* p)
    # fxλp_norm = norm(fx, 2)
    # TODO: nan backtrack

    j = 0

    f!(fx, x .+ λ₂ .* p)
    converged = _test_approximate_norm_descent_convergence(
        f!,
        fx,
        x,
        fx_norm,
        p,
        σ₁,
        λ₂,
        η,
    )

    while j < max_iter && !converged
        j += 1
        λ₁, λ₂ = λ₂, β * λ₂
        converged = _test_approximate_norm_descent_convergence(
            f!,
            fx,
            x,
            fx_norm,
            p,
            σ₁,
            λ₂,
            η,
        )
    end

    return λ₂
end


function _test_approximate_norm_descent_convergence(
    f!,
    fx,
    x,
    fx_norm,
    p,
    σ₁,
    λ₂,
    η,
)
    f!(fx, x .+ λ₂ .* p)
    n1 = norm(fx, 2)

    f!(fx, x)
    n2 = norm(fx, 2)

    return n1 ≤ fx_norm - σ₁ * norm(λ₂ .* p, 2) .^ 2 + η * n2
end