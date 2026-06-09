using Pkg
using SafeTestsets, Test

const GROUP = uppercase(get(ENV, "GROUP", "CORE"))

@info "Running tests for GROUP: $GROUP"

@time begin
    if GROUP == "CORE" || GROUP == "CPU" || GROUP == "ALL"
        @time @safetestset "Utils Tests" include("utils_tests.jl")
        @time @safetestset "Layers Tests" include("layers_tests.jl")
    end

    if GROUP == "QA"
        Pkg.activate(joinpath(@__DIR__, "qa"))
        Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
        Pkg.instantiate()
        @time @safetestset "Quality Assurance Tests" include(joinpath("qa", "qa.jl"))
    end
end
