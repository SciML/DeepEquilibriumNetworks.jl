# Fixed Point Iteration Solvers
## Why reimplement this?
## 1. NLsolve doesn't play well with GPU Code
## 2. NLsolve also assumes Matrices while in DL the standard is 4D Arrays

struct AndersonAccelerationCache{X,F,H,y,R,Z}
    X_cache::X
    F_cache::F
    H_cache::H
    y_cache::y
    residuals::R
    input_dims::Z
end

function AndersonAccelerationCache(init_x, m::Int, max_iterations::Int)
    W, H, C, B = size(init_x)

    X = init_zero(init_x, (W * H * C, m, B))
    F = init_zero(init_x, (W * H * C, m, B))

    H = init_zero(init_x, (m + 1, m + 1, B))
    H[2:end, 1, :] .= 1
    H[1, 2:end, :] .= 1

    y = init_zero(init_x, (1, m + 1, B))
    y[:, 1, :] .= 1

    residuals = Vector{eltype(init_x)}(undef, max_iterations - 1)

    return AndersonAccelerationCache(X, F, H, y, residuals, size(init_x))
end


mutable struct AndersonAcceleration{T,C}
    m::Int
    max_iterations::Int
    λ::T
    η::T
    β::T
    cache::C
end


function maybe_reinit_cache!(adacc::AndersonAcceleration, init_x)
    if size(init_x) != adacc.cache.input_dims
        adacc.cache =
            AndersonAccelerationCache(init_x, adacc.m, adacc.max_iterations)
    end
    return adacc
end


function AndersonAcceleration(
    init_x::AbstractArray{T,4};
    m::Int = 5,
    max_iterations::Int = 50,
    λ::T = 1f-4,
    η::T = 1f-3,
    β::T = 1.0f0,
) where {T}
    return AndersonAcceleration(
        m,
        max_iterations,
        λ,
        η,
        β,
        AndersonAccelerationCache(init_x, m, max_iterations),
    )
end


function fixedpointsolve(method::AndersonAcceleration, f, init_x)
    method = maybe_reinit_cache!(method, init_x)

    W, H, C, B = size(init_x)

    f₀ = f(init_x)
    f₁ = f(f₀)

    X, F, H, y = (
        method.cache.X_cache,
        method.cache.F_cache,
        method.cache.H_cache,
        method.cache.y_cache,
    )

    X[:, 1, :] .= reshape(init_x, :, B)
    F[:, 1, :] .= reshape(f₀, :, B)

    X[:, 2, :] .= F[:, 1, :]
    F[:, 2, :] .= reshape(f₁, :, B)

    residuals = method.cache.residuals
    total_iterations = 0

    for k = 2:method.max_iterations
        n = min(k, method.m)
        idx = mod1(k + 1, method.m)

        G = F[:, 1:n, :] .- X[:, 1:n, :]

        H[2:n+1, 2:n+1, :] .= (
            batched_mul(permutedims(G, (2, 1, 3)), G) .+
            method.λ .* reshape(init_identity_matrix(init_x, n), (n, n, 1))
        )

        α = cat(
            eachslice(H[1:n+1, 1:n+1, :]; dims = 3) .\
            transpose.(eachslice(y[:, 1:n+1, :]; dims = 3))...;
            dims = 3,
        )[
            2:n+1,
            1:1,
            :,
        ]

        if isone(method.β)
            X[:, idx:idx, :] .= batched_mul(F[:, 1:n, :], α)
        elseif iszero(method.β)
            X[:, idx:idx, :] .= batched_mul(X[:, 1:n, :], α)
        else
            X[:, idx:idx, :] .=
                method.β .* batched_mul(F[:, 1:n, :], α) .+
                (1 .- method.β) .* batched_mul(X[:, 1:n, :], α)
        end

        F[:, idx, :] .= reshape(f(reshape(X[:, idx, :], size(init_x))), :, B)

        residuals[k-1] =
            norm(F[:, idx, :] .- X[:, idx, :]) ./
            (eps(eltype(init_x)) + norm(F[:, idx, :]))

        residuals[k-1] < method.η && (total_iterations = k; break)
    end

    return reshape(X[:, mod1(total_iterations, method.m), :], size(init_x))
end
