# Limited Memory Broyden
"""
    LimitedMemoryBroydenSolver(; T=Float32, device, original_dims::Tuple{Int,Int}, batch_size, maxiters::Int=50,
                               ϵ::Real=1e-6, criteria::Symbol=:reltol, abstol::Union{Real,Nothing}=nothing,
                               reltol::Union{Real,Nothing}=nothing

Limited Memory Broyden Solver ([baimultiscale2020](@cite)) for solving Discrete DEQs.

## Arguments

  - `T`: The type of the elements of the vectors. (Default: `Float32`)
  - `device`: The device to use. Pass `gpu` to use the GPU else pass `cpu`.
  - `original_dims`: Dimensions to reshape the arrays into (excluding the batch dimension).
  - `batch_size`: The batch size of the problem. Your inputs can have a different batch size, but having
    them match allows us to efficiently cache internal statistics without reallocation.
  - `maxiters`: Maximum number of iterations to run.
  - `ϵ`: Tolerance for convergence.
  - `criteria`: The criteria to use for convergence. Can be `:reltol` or `:abstol`.
  - `abstol`: Absolute tolerance.
  - `reltol`: Relative tolerance.

See also: [`BroydenSolver`](@ref)
"""
struct LimitedMemoryBroydenSolver end

@inbounds @views function nlsolve(::LimitedMemoryBroydenSolver, f::Function,
                                  y::AbstractMatrix{T}; terminate_condition,
                                  maxiters::Int=10) where {T}
  LBFGS_threshold = min(maxiters, 27)

  total_hsize, batch_size = size(y)

  # Initialize the cache
  x₀ = copy(y)
  fx₀ = f(x₀)
  x₁ = copy(y)
  Δx = copy(x₀)
  Δfx = copy(x₀)
  Us = fill!(similar(y, (LBFGS_threshold, total_hsize, batch_size)), T(0))
  VTs = fill!(similar(y, (total_hsize, LBFGS_threshold, batch_size)), T(0))

  # Store the trajectory
  xs = [x₀]

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

    push!(xs, x₁)

    # Convergence Check
    terminate_condition(fx₁, x₁) && break

    # Compute the update
    part_Us = Us[1:min(LBFGS_threshold, nstep), :, :]
    part_VTs = VTs[:, 1:min(LBFGS_threshold, nstep), :]

    vT = rmatvec(part_Us, part_VTs, Δx)  # D x C x N
    mvec = matvec(part_Us, part_VTs, Δfx)
    vTΔfx = sum(vT .* Δfx; dims=(1, 2))
    @. Δx = (Δx - mvec) / (vTΔfx + eps(T))  # D x C x N

    VTs[:, mod1(nstep, LBFGS_threshold), :] .= vT
    Us[mod1(nstep, LBFGS_threshold), :, :] .= Δx

    update = -matvec(Us[1:min(LBFGS_threshold, nstep + 1), :, :],
                     VTs[:, 1:min(LBFGS_threshold, nstep + 1), :], fx₁)
    copyto!(x₀, x₁)
    copyto!(fx₀, fx₁)

    # Increment Counter
    nstep += 1
  end

  return xs, (nf=nstep + 1,)
end

@inbounds @views function matvec(part_Us::AbstractArray{E, 3},
                                 part_VTs::AbstractArray{E, 3},
                                 x::AbstractArray{E, 2}) where {E}
  # part_Us -> (T x D x N) | part_VTs -> (D x T x N) | x -> (D x N)
  xTU = sum(unsqueeze(x; dims=1) .* part_Us; dims=2) # T x 1 x N
  return -x .+ dropdims(sum(permutedims(xTU, (2, 1, 3)) .* part_VTs; dims=2); dims=2)
end

@inbounds @views function rmatvec(part_Us::AbstractArray{E, 3},
                                  part_VTs::AbstractArray{E, 3},
                                  x::AbstractArray{E, 2}) where {E}
  # part_Us -> (T x D x N) | part_VTs -> (D x T x N) | x -> (D x N)
  VTx = sum(part_VTs .* unsqueeze(x; dims=2); dims=1) # 1 x T x N
  return -x .+ dropdims(sum(part_Us .* permutedims(VTx, (2, 1, 3)); dims=1); dims=1)
end
