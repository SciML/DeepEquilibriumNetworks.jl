import LinearAlgebra

"""
    BroydenSolver()

Broyden Solver ([broyden1965class](@cite)) for solving Discrete DEQs. It is recommended to
use [`LimitedMemoryBroydenSolver`](@ref) for better performance.

See also: [`LimitedMemoryBroydenSolver`](@ref)
"""
struct BroydenSolver end

function nlsolve(b::BroydenSolver, f::Function, y::AbstractArray{T}; terminate_condition,
                 maxiters::Int=10) where {T}
  res, stats = nlsolve(b, u -> vec(f(reshape(u, size(y)))), vec(y); terminate_condition,
                       maxiters)
  return reshape.(res, (size(y),)), stats
end

function nlsolve(::BroydenSolver, f::Function, y::AbstractVector{T}; terminate_condition,
                 maxiters::Int=10) where {T}
  x, x_old, dx, fx_old = copy(y), copy(y), copy(y), f(y)
  dfx = copy(fx_old)
  Jinv = init_identity_matrix(x)
  p = similar(fx_old, (size(Jinv, 1),))
  rho, sigma_2 = T(0.9), T(0.001)

  # Store the trajectory
  xs = [copy(x)]

  maybe_stuck, max_resets, resets, nsteps, nf = false, 3, 0, 1, 1

  while nsteps <= maxiters
    LinearAlgebra.mul!(p, Jinv, fx_old)
    p .*= -1

    x .= x_old .+ p
    fx = f(x)
    nf += 1

    if (LinearAlgebra.norm(fx, 2) ≤
        rho * LinearAlgebra.norm(fx_old, 2) - sigma_2 * LinearAlgebra.norm(p, 2)^2)
      alpha = T(1)
    else
      alpha, _stats = _approximate_norm_descent(f, x, p)
      x .= x_old .+ alpha .* p
      fx = f(x)
      nf += 1 + _stats.nf
    end

    dx .= x .- x_old
    dfx .= fx .- fx_old

    maybe_stuck = all(abs.(dx) .<= eps(T)) || all(abs.(dfx) .<= eps(T))
    if maybe_stuck
      Jinv = init_identity_matrix(x)
      resets += 1
      maybe_stuck = (resets ≤ max_resets) && maybe_stuck
    else
      dxJinv = dx' * Jinv
      Jinv .+= ((dx .- Jinv * dfx) ./ (dxJinv * dfx)) * dxJinv
    end

    maybe_stuck = false
    nsteps += 1
    copyto!(fx_old, fx)
    copyto!(x_old, x)

    push!(xs, copy(x))

    # Convergence Check
    terminate_condition(fx, x) && break
  end

  return xs, (nf=nf,)
end

function _approximate_norm_descent(f::Function, x::AbstractArray{T, N}, p; lambda_0=T(1),
                                   beta=T(0.5), sigma_1=T(0.001), eta=T(0.1),
                                   max_iter=50) where {T, N}
  lambda_2, lambda_1 = lambda_0, lambda_0

  fx = f(x)
  fx_norm = LinearAlgebra.norm(fx, 2)
  j = 1
  fx = f(x .+ lambda_2 .* p)
  converged = false

  while j <= max_iter && !converged
    j += 1
    lambda_1, lambda_2 = lambda_2, beta * lambda_2
    converged = _test_approximate_norm_descent_convergence(f, x, fx_norm, p, sigma_1,
                                                           lambda_2, eta)
  end

  return lambda_2, (nf=2 * (j + 1),)
end

function _test_approximate_norm_descent_convergence(f, x, fx_norm, p, sigma_1, lambda_2,
                                                    eta)
  n1 = LinearAlgebra.norm(f(x .+ lambda_2 .* p), 2)
  n2 = LinearAlgebra.norm(f(x), 2)
  return n1 ≤ fx_norm - sigma_1 * LinearAlgebra.norm(lambda_2 .* p, 2) .^ 2 + eta * n2
end
