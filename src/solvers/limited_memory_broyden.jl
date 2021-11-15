# Limited Memory Broyden
## Reference: https://arxiv.org/pdf/2006.08656.pdf
# struct LimitedMemoryBroydenCache{bT,uT,vT,F,X}
#     B::bT
#     u::uT
#     v::vT
#     fx::F
#     Δfx::F
#     fx_old::F
#     x::X
#     Δx::X
#     x_old::X
# end

# function LimitedMemoryBroydenCache(x)
#     fx, Δfx, fx_old = copy(x), copy(x), copy(x)
#     x, Δx, x_old = copy(x), copy(x), copy(x)
#     Jinv = _init_identity_matrix(x)
#     return LimitedMemoryBroydenCache(Jinv, fx, Δfx, fx_old, x, Δx, x_old)
# end

function line_search(update, x₀, f₀, f, nstep::Int = 0, on::Bool = false)
    # TODO: Implement a line search algorithm
    x_est = x₀ .+ update
    f₀_new = f(x_est)
    return (x_est, f₀_new, x_est .- x₀, f₀_new .- f₀, 0)
end

# This function is designed for the case where x_actual is a 4D Array (Images)
function limited_memory_broyden(
    f!::Function,
    x_::AbstractVector{T};
    # If size(x_actual) = (D x D x C x N) then original_dims = (2D x C)
    original_dims::Tuple{Int,Int},
    maxiters::Int = 10,
    ϵ::Real = 1e-6,
    abstol::Union{Real,Nothing} = nothing,
    reltol::Union{Real,Nothing} = nothing,
) where {T}
    ϵ = abstol !== nothing ? abstol : ϵ

    if reltol !== nothing
        @warn maxlog = 1 "reltol is set to $reltol, but `limited_memory_broyden` ignores this value"
    end

    nstep = 1
    tnstep = 1
    LBFGS_threshold = min(maxiters, 27)

    # Make allocations
    ## TODO: Use a cache to prevent recurring allocations
    x__ = copy(x_)
    x_est = reshape(x__, original_dims..., :)  # 2D x C x N
    total_hsize, n_elem, batch_size = size(x_est)
    actual_size = (total_hsize, n_elem, batch_size)

    fx_ = zero(x_)  # ((2D x C x N),)
    f(x) = (f!(fx_, vec(x)); return reshape(fx_, actual_size))
    fx = f(x_est)  # 2D x C x N

    # L x 2D x C x N
    Us = fill!(
        similar(x_est, (LBFGS_threshold, total_hsize, n_elem, batch_size)),
        T(0),
    )
    # 2D x C x L x N
    VTs = fill!(
        similar(x_est, (total_hsize, n_elem, LBFGS_threshold, batch_size)),
        T(0),
    )

    protect_threshold = T(1e6) * n_elem
    update = reshape(fx, actual_size)
    new_objective = norm(fx)
    initial_objective = new_objective
    lowest_objective = new_objective
    lowest_xest = x_est

    @inbounds while nstep < maxiters
        x_est, fx, Δx, Δfx, ite =
            line_search(update, x_est, fx, f, nstep, false)
        nstep += 1
        tnstep += (ite + 1)

        new_objective = norm(fx)
        # TODO: Terminate Early if Stagnant
        if new_objective < lowest_objective
            lowest_objective = new_objective
            lowest_xest = x_est
        end
        new_objective < ϵ && break

        # Prevent Divergence
        (new_objective > initial_objective * protect_threshold) && break

        @views part_Us = Us[1:min(LBFGS_threshold, nstep), :, :, :]
        @views part_VTs = VTs[:, :, 1:min(LBFGS_threshold, nstep), :]

        vT = rmatvec(part_Us, part_VTs, Δx)  # 2D x C x N
        u =
            (Δx .- matvec(part_Us, part_VTs, Δfx)) ./
            sum(vT .* Δfx, dims = (1, 2))  # 2D x C x N
        vT[.!isfinite.(vT)] .= T(0)
        u[.!isfinite.(u)] .= T(0)

        @views VTs[:, :, mod1(nstep, LBFGS_threshold), :] .= vT
        @views Us[mod1(nstep, LBFGS_threshold), :, :, :] .= u

        @views update =
            -matvec(
                part_Us[1:min(LBFGS_threshold, nstep + 1), :, :, :],
                VTs[:, :, 1:min(LBFGS_threshold, nstep + 1), :],
                fx,
            )
    end

    return vec(lowest_xest)
end


function matvec(
    part_Us::AbstractArray{E,4},
    part_VTs::AbstractArray{E,4},
    x::AbstractArray{E,3},
) where {E}
    # part_Us -> (T x D x C x N)
    # part_VTs -> (D x C x T x N)
    # x -> (D x C x N)
    length(part_Us) == 0 && return -x
    T, D, C, N = size(part_Us)
    xTU = sum(reshape(x, (1, D, C, N)) .* part_Us, dims = (2, 3)) # T x 1 x 1 x N
    return -x .+ reshape(
        sum(permutedims(xTU, (2, 3, 1, 4)) .* part_VTs, dims = 3),
        (D, C, N),
    )
end

function rmatvec(
    part_Us::AbstractArray{E,4},
    part_VTs::AbstractArray{E,4},
    x::AbstractArray{E,3},
) where {E}
    # part_Us -> (T x D x C x N)
    # part_VTs -> (D x C x T x N)
    # x -> (D x C x N)
    length(part_Us) == 0 && return -x
    T, D, C, N = size(part_Us)
    VTx = sum(part_VTs .* reshape(x, (D, C, 1, N)), dims = (1, 2)) # 1 x 1 x T x N
    return -x .+ reshape(
        sum(part_Us .* permutedims(VTx, (3, 1, 2, 4)), dims = 1),
        (D, C, N),
    )
end
