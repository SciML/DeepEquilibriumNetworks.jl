"""
    BroydenSolver(; T=Float32, device, original_dims::Tuple{Int,Int}, batch_size, maxiters::Int=50, ϵ::Real=1e-6,
                  abstol::Union{Real,Nothing}=nothing, reltol::Union{Real,Nothing}=nothing)

Broyden Solver ([broyden1965class](@cite)) for solving Discrete DEQs. It is recommended to use [`LimitedMemoryBroydenSolver`](@ref) for better performance.

## Arguments

  - `T`: The type of the elements of the vectors. (Default: `Float32`)
  - `device`: The device to use. Pass `gpu` to use the GPU else pass `cpu`.
  - `original_dims`: Dimensions to reshape the arrays into (excluding the batch dimension).
  - `batch_size`: The batch size of the problem. Your inputs can have a different batch size, but having
    them match allows us to efficiently cache internal statistics without reallocation.
  - `maxiters`: Maximum number of iterations to run.
  - `ϵ`: Tolerance for convergence.
  - `abstol`: Absolute tolerance.
  - `reltol`: Relative tolerance. (This value is ignored by `BroydenSolver` at the moment)

See also: [`LimitedMemoryBroydenSolver`](@ref)
"""
struct BroydenSolver end

function nlsolve(b::BroydenSolver, f::Function, y::AbstractArray{T}; terminate_condition,
                 maxiters::Int=10) where {T}
  res, stats = nlsolve(b,
                       u -> vec(f(reshape(u, size(y)))),
                       vec(y);
                       terminate_condition,
                       maxiters)
  return reshape.(res, (size(y),)), stats
end

function nlsolve(::BroydenSolver, f::Function, y::AbstractVector{T}; terminate_condition,
                 maxiters::Int=10) where {T}
  x = copy(y)
  x_old = copy(y)
  Δx = copy(y)
  fx_old = f(y)
  Δfx = copy(fx_old)
  Jinv = _init_identity_matrix(y)
  p = similar(fx_old, (size(Jinv, 1),))
  ρ, σ₂ = T(0.9), T(0.001)

  # Store the trajectory
  xs = [x]

  maybe_stuck, max_resets, resets, nsteps, nf = false, 3, 0, 1, 1

  while nsteps <= maxiters
    mul!(p, Jinv, fx_old)
    p .*= -1

    @. x = x_old + p
    fx = f(x)
    nf += 1

    if norm(fx, 2) ≤ ρ * norm(fx_old, 2) - σ₂ * norm(p, 2)^2
      α = T(1)
    else
      α, _stats = _approximate_norm_descent(f, x, p)
      @. x = x_old + α * p
      fx = f(x)
      nf += 1 + _stats.nf
    end

    @. Δx = x - x_old
    @. Δfx = fx - fx_old

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
    nsteps += 1
    copyto!(fx_old, fx)
    copyto!(x_old, x)

    push!(xs, x)

    # Convergence Check
    terminate_condition(fx, x) && break
  end

  return xs, (nf=nf,)
end

function _approximate_norm_descent(f::Function, x::AbstractArray{T, N}, p; λ₀=T(1),
                                   β=T(0.5), σ₁=T(0.001),
                                   η=T(0.1), max_iter=50) where {T, N}
  λ₂, λ₁ = λ₀, λ₀

  fx = f(x)
  fx_norm = norm(fx, 2)
  j = 1
  fx = f(x .+ λ₂ .* p)
  converged = false

  while j <= max_iter && !converged
    j += 1
    λ₁, λ₂ = λ₂, β * λ₂
    converged = _test_approximate_norm_descent_convergence(f, x, fx_norm, p, σ₁, λ₂, η)
  end

  return λ₂, (nf=2(j + 1),)
end

function _test_approximate_norm_descent_convergence(f, x, fx_norm, p, σ₁, λ₂, η)
  n1 = norm(f(x .+ λ₂ .* p), 2)
  n2 = norm(f(x), 2)
  return n1 ≤ fx_norm - σ₁ * norm(λ₂ .* p, 2) .^ 2 + η * n2
end
