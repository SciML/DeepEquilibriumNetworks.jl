# Limited Memory Broyden
"""
    LimitedMemoryBroydenSolver(; T=Float32, device, original_dims::Tuple{Int,Int}, batch_size, maxiters::Int=50,
                               ϵ::Real=1e-6, criteria::Symbol=:reltol, abstol::Union{Real,Nothing}=nothing,
                               reltol::Union{Real,Nothing}=nothing

Limited Memory Broyden Solver ([baimultiscale2020](@cite)) for solving Discrete DEQs.

## Arguments

* `T`: The type of the elements of the vectors. (Default: `Float32`)
* `device`: The device to use. Pass `gpu` to use the GPU else pass `cpu`.
* `original_dims`: Dimensions to reshape the arrays into (excluding the batch dimension).
* `batch_size`: The batch size of the problem. Your inputs can have a different batch size, but having
                them match allows us to efficiently cache internal statistics without reallocation.
* `maxiters`: Maximum number of iterations to run.
* `ϵ`: Tolerance for convergence.
* `criteria`: The criteria to use for convergence. Can be `:reltol` or `:abstol`.
* `abstol`: Absolute tolerance.
* `reltol`: Relative tolerance.

See also: [`BroydenSolver`](@ref)
"""
struct LimitedMemoryBroydenSolver end

function nlsolve(l::LimitedMemoryBroydenSolver, f::Function, y::AbstractMatrix; kwargs...)
    res, stats = nlsolve(l, f, reshape(y, size(y, 1), 1, size(y, 2)); kwargs...)
    return dropdims(res; dims=2), stats
end

function nlsolve(
    ::LimitedMemoryBroydenSolver, f::Function, y::AbstractArray{T,3}; terminate_condition, maxiters::Int=10
) where {T}
    LBFGS_threshold = min(maxiters, 27)

    total_hsize, n_elem, batch_size = size(y)

    # Initialize the cache
    x₀ = copy(y)
    fx₀ = f(x₀)
    x₁ = copy(y)
    Δx = copy(x₀)
    Δfx = copy(x₀)
    Us = fill!(similar(y, (LBFGS_threshold, total_hsize, n_elem, batch_size)), T(0))
    VTs = fill!(similar(y, (total_hsize, n_elem, LBFGS_threshold, batch_size)), T(0))

    # Counters
    nstep = 1

    # Main Algorithm
    update = fx₀

    while nstep <= maxiters
        # Update
        @. x₁ = x₀ + update
        fx₁ = f(x₁)
        @. Δx = x₁ - x₀
        @. Δfx = fx₁ - fx₀

        # Convergence Check
        terminate_condition(fx₁, x₁) && break

        # Compute the update
        @views part_Us = Us[1:min(LBFGS_threshold, nstep), :, :, :]
        @views part_VTs = VTs[:, :, 1:min(LBFGS_threshold, nstep), :]

        vT = rmatvec(part_Us, part_VTs, Δx)  # D x C x N
        mvec = matvec(part_Us, part_VTs, Δfx)
        vTΔfx = sum(vT .* Δfx; dims=(1, 2))
        @. Δx = (Δx - mvec) / (vTΔfx + eps(T))  # D x C x N

        @views VTs[:, :, mod1(nstep, LBFGS_threshold), :] .= vT
        @views Us[mod1(nstep, LBFGS_threshold), :, :, :] .= Δx

        @views update =
            -matvec(
                Us[1:min(LBFGS_threshold, nstep + 1), :, :, :], VTs[:, :, 1:min(LBFGS_threshold, nstep + 1), :], fx₁
            )
        copyto!(x₀, x₁)
        copyto!(fx₀, fx₁)

        # Increment Counter
        nstep += 1
    end

    return x₁, (nf=nstep + 1,)
end

@inbounds function matvec(part_Us::AbstractArray{E,4}, part_VTs::AbstractArray{E,4}, x::AbstractArray{E,3}) where {E}
    # part_Us -> (T x D x C x N) | part_VTs -> (D x C x T x N) | x -> (D x C x N)
    _, D, C, N = size(part_Us)
    xTU = sum(reshape(x, (1, D, C, N)) .* part_Us; dims=(2, 3)) # T x 1 x 1 x N
    return -x .+ dropdims(sum(permutedims(xTU, (2, 3, 1, 4)) .* part_VTs; dims=3); dims=3)
end

function rmatvec(part_Us::AbstractArray{E,4}, part_VTs::AbstractArray{E,4}, x::AbstractArray{E,3}) where {E}
    # part_Us -> (T x D x C x N) | part_VTs -> (D x C x T x N) | x -> (D x C x N)
    _, D, C, N = size(part_Us)
    VTx = sum(part_VTs .* reshape(x, (D, C, 1, N)); dims=(1, 2)) # 1 x 1 x T x N
    return -x .+ dropdims(sum(part_Us .* permutedims(VTx, (3, 1, 2, 4)); dims=1); dims=1)
end
