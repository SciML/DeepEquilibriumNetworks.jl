# Dynamical System Variants

[baideep2019](@cite) introduced Discrete Deep Equilibrium Models which drives a Discrete Dynamical System to its steady-state. [pal2022mixing](@cite) extends this framework to Continuous Dynamical Systems which converge to the steady-stable in a more stable fashion. For a detailed discussion refer to [pal2022mixing](@cite).

## Continuous DEQs

```@docs
ContinuousDEQSolver
```

## Discrete DEQs

```@docs
DiscreteDEQSolver
```

## Termination Conditions

#### Termination on Absolute Tolerance

  - `:abs`: Terminates if ``all \left( | \frac{\partial u}{\partial t} | \leq abstol \right)``
  - `:abs_norm`: Terminates if ``\| \frac{\partial u}{\partial t} \| \leq abstol``
  - `:abs_deq_default`: Essentially `abs_norm` + terminate if there has been no improvement for the last 30 steps + terminate if the solution blows up (diverges)
  - `:abs_deq_best`: Same as `:abs_deq_default`, but uses the best solution found so far, i.e. deviates only if the solution has not converged

#### Termination on Relative Tolerance

  - `:rel`: Terminates if ``all \left(| \frac{\partial u}{\partial t} | \leq reltol \times | u | \right)``
  - `:rel_norm`: Terminates if ``\| \frac{\partial u}{\partial t} \| \leq reltol \times \| \frac{\partial u}{\partial t} + u \|``
  - `:rel_deq_default`: Essentially `rel_norm` + terminate if there has been no improvement for the last 30 steps + terminate if the solution blows up (diverges)
  - `:rel_deq_best`: Same as `:rel_deq_default`, but uses the best solution found so far, i.e. deviates only if the solution has not converged

#### Termination using both Absolute and Relative Tolerances

  - `:norm`: Terminates if ``\| \frac{\partial u}{\partial t} \| \leq reltol \times \| \frac{\partial u}{\partial t} + u \|`` &
    ``\| \frac{\partial u}{\partial t} \| \leq abstol``
  - `fallback`: Check if all values of the derivative are close to zero wrt both relative and absolute tolerance. This is usable for small problems
    but doesn't scale well for neural networks, and should be avoided unless absolutely necessary
