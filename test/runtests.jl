using SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "ALL")

if GROUP == "ALL" || GROUP == "CORE"
  @safetestset "Solve" begin include("solve.jl") end
  @safetestset "Utilities" begin include("utils.jl") end
end

if GROUP == "ALL" || GROUP == "LAYERS"
  @safetestset "DEQ Core" begin include("layers/core.jl") end
  @safetestset "Jacobian Regularization" begin include("layers/jacobian_stabilization.jl") end
  @safetestset "DEQ" begin include("layers/deq.jl") end
  @safetestset "Multiscale DEQ" begin include("layers/mdeq.jl") end
end

if GROUP == "ALL" || GROUP == "ADJOINT"
  @safetestset "Adjoint" begin include("adjoint.jl") end
end
