# Limited Memory Broyden
## Reference: https://arxiv.org/pdf/2006.08656.pdf
struct LimitedMemoryBroydenCache{uT,vT,F,X}
    Us::uT
    VTs::vT
    fx_::F
    x::X
end

struct LimitedMemoryBroydenSolver{C<:LimitedMemoryBroydenCache,T<:Real}
    cache::C
    original_dims::Tuple{Int,Int}
    maxiters::Int
    batch_size::Int
    ϵ::T
end

function LimitedMemoryBroydenSolver(
    T = Float32;
    device,
    original_dims::Tuple{Int,Int},
    batch_size,
    maxiters::Int = 50,
    ϵ::Real = 1e-6,
    abstol::Union{Real,Nothing} = nothing,
    reltol::Union{Real,Nothing} = nothing,
)
    ϵ = abstol !== nothing ? abstol : ϵ

    if reltol !== nothing
        @warn maxlog = 1 "reltol is set to $reltol, but `LimitedMemoryBroydenSolver` ignores this value"
    end

    LBFGS_threshold = min(maxiters, 27)

    x = zeros(T, original_dims..., batch_size) |> device
    fx = zeros(T, original_dims..., batch_size) |> device

    total_hsize, n_elem, batch_size = size(x)

    # L x 2D x C x N
    Us = fill!(
        similar(x, (LBFGS_threshold, total_hsize, n_elem, batch_size)),
        T(0),
    )
    # 2D x C x L x N
    VTs = fill!(
        similar(x, (total_hsize, n_elem, LBFGS_threshold, batch_size)),
        T(0),
    )

    cache = LimitedMemoryBroydenCache(Us, VTs, vec(fx), x)

    return LimitedMemoryBroydenSolver(
        cache,
        original_dims,
        maxiters,
        batch_size,
        T(ϵ),
    )
end

function line_search(update, x₀, f₀, f, nstep::Int = 0, on::Bool = false)
    # TODO: Implement a line search algorithm
    x_est = x₀ .+ update
    f₀_new = f(x_est)
    return (x_est, f₀_new, x_est .- x₀, f₀_new .- f₀, 0)
end

function (lbroyden::LimitedMemoryBroydenSolver{C,T})(f!, x_::AbstractVector{T}) where {C,T}
    @unpack cache, original_dims, batch_size, maxiters, ϵ = lbroyden
    nfeatures = prod(original_dims)
    if nfeatures * batch_size != length(x_)
        # Maybe the last batch is smaller than the others
        cache = LimitedMemoryBroydenSolver(
            T;
            device = x_ isa CuArray ? gpu : cpu,
            original_dims = original_dims,
            batch_size = length(x_) ÷ nfeatures,
            maxiters = maxiters,
            ϵ = ϵ,
        ).cache
    end
    @unpack Us, VTs, fx_, x = cache
    x .= reshape(x_, size(x))
    LBFGS_threshold = size(Us, 1)
    fill!(Us, T(0))
    fill!(VTs, T(0))

    # Counters
    nstep = 1
    tnstep = 1

    # Initialize
    total_hsize, n_elem, batch_size = size(x)
    actual_size = size(x)

    # Modify the functions
    f(x) = (f!(fx_, vec(x)); return reshape(fx_, actual_size))
    fx = f(x)

    update = fx
    new_objective = norm(fx)

    protect_threshold = T(1e6) * n_elem
    initial_objective = new_objective
    lowest_objective = new_objective
    lowest_xest = x

    @inbounds while nstep < maxiters
        x, fx, Δx, Δfx, ite =
            line_search(update, x, fx, f, nstep, false)
        nstep += 1
        tnstep += (ite + 1)

        new_objective = norm(fx)
        # TODO: Terminate Early if Stagnant
        if new_objective < lowest_objective
            lowest_objective = new_objective
            lowest_xest = x
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
                Us[1:min(LBFGS_threshold, nstep + 1), :, :, :],
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
