import SafeTestsets
import Test

const GROUP = get(ENV, "GROUP", "ALL")

if GROUP == "ALL" || GROUP == "CORE"
  SafeTestsets.@safetestset "Operator" begin include("operator.jl") end
  SafeTestsets.@safetestset "Discrete Solvers: Broyden" begin include("solvers/discrete/broyden.jl") end
  SafeTestsets.@safetestset "Discrete Solvers: L-Broyden" begin include("solvers/discrete/limited_memory_broyden.jl") end
  SafeTestsets.@safetestset "Solve" begin include("solve.jl") end
  SafeTestsets.@safetestset "Utilities" begin include("utils.jl") end
end

if GROUP == "ALL" || GROUP == "LAYERS"
  SafeTestsets.@safetestset "DEQ Core" begin include("layers/core.jl") end
  SafeTestsets.@safetestset "Jacobian Regularization" begin include("layers/jacobian_stabilization.jl") end
  SafeTestsets.@safetestset "DEQ" begin include("layers/deq.jl") end
  SafeTestsets.@safetestset "Multiscale DEQ" begin include("layers/mdeq.jl") end
end

if GROUP == "ALL" || GROUP == "ADJOINT"
  SafeTestsets.@safetestset "Adjoint" begin include("adjoint.jl") end
end
