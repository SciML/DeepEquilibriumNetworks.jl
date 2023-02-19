import MLUtils

# Limited Memory Broyden
"""
    LimitedMemoryBroydenSolver()

Limited Memory Broyden Solver ([baimultiscale2020](@cite)) for solving Discrete DEQs.

See also: [`BroydenSolver`](@ref)
"""
struct LimitedMemoryBroydenSolver end

@inbounds function nlsolve(::LimitedMemoryBroydenSolver, f::Function, y::AbstractMatrix{T};
                           terminate_condition, maxiters::Int, abstol, reltol) where {T}
  LBFGS_threshold = min(maxiters, 27)

  total_hsize, batch_size = size(y)

  # Initialize the cache
  x0 = copy(y)
  fx0 = f(x0)
  x1 = copy(y)
  dx = copy(x0)
  dfx = copy(x0)
  Us = fill!(similar(y, (LBFGS_threshold, total_hsize, batch_size)), T(0))
  VTs = fill!(similar(y, (total_hsize, LBFGS_threshold, batch_size)), T(0))

  # Store the trajectory
  xs = [copy(x0)]

  # Counters
  nstep = 1

  # Main Algorithm
  update = fx0

  while nstep <= maxiters
    # Update
    x1 .= x0 .+ update
    fx1 = f(x1)
    dx .= x1 - x0
    dfx .= fx1 - fx0

    push!(xs, copy(x1))

    # Convergence Check
    terminate_condition(fx1, x1, xs[end - 1], abstol, reltol) && break

    # Compute the update
    part_Us = view(Us, 1:min(LBFGS_threshold, nstep), :, :)
    part_VTs = view(VTs, :, 1:min(LBFGS_threshold, nstep), :)

    vT = _rmatvec(part_Us, part_VTs, dx)  # D x C x N
    mvec = _matvec(part_Us, part_VTs, dfx)
    vTdfx = sum(vT .* dfx; dims=(1, 2))
    dx .= (dx .- mvec) ./ (vTdfx .+ eps(T))  # D x C x N

    view(VTs, :, mod1(nstep, LBFGS_threshold), :) .= vT
    view(Us, mod1(nstep, LBFGS_threshold), :, :) .= dx

    update = -_matvec(view(Us, 1:min(LBFGS_threshold, nstep + 1), :, :),
                      view(VTs, :, 1:min(LBFGS_threshold, nstep + 1), :), fx1)
    copyto!(x0, x1)
    copyto!(fx0, fx1)

    # Increment Counter
    nstep += 1
  end

  return xs, (nf=nstep + 1,)
end

@inbounds function _matvec(part_Us::AbstractArray{E, 3}, part_VTs::AbstractArray{E, 3},
                           x::AbstractArray{E, 2}) where {E}
  # part_Us -> (T x D x N) | part_VTs -> (D x T x N) | x -> (D x N)
  xTU = sum(MLUtils.unsqueeze(x; dims=1) .* part_Us; dims=2) # T x 1 x N
  return -x .+ dropdims(sum(permutedims(xTU, (2, 1, 3)) .* part_VTs; dims=2); dims=2)
end

@inbounds function _rmatvec(part_Us::AbstractArray{E, 3}, part_VTs::AbstractArray{E, 3},
                            x::AbstractArray{E, 2}) where {E}
  # part_Us -> (T x D x N) | part_VTs -> (D x T x N) | x -> (D x N)
  VTx = sum(part_VTs .* MLUtils.unsqueeze(x; dims=2); dims=1) # 1 x T x N
  return -x .+ dropdims(sum(part_Us .* permutedims(VTx, (2, 1, 3)); dims=1); dims=1)
end
