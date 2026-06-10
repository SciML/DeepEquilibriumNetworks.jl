using Pkg
using SafeTestsets, Test

const GROUP = uppercase(get(ENV, "GROUP", "CORE"))

@info "Running tests for GROUP: $GROUP"

# GPU is the self-hosted CUDA runner cell of test/test_groups.toml: the same
# functional suite, with shared_testsetup.jl's backend switched to CUDA.
GROUP == "GPU" && (ENV["BACKEND_GROUP"] = "CUDA")

@time begin
    if GROUP == "CORE" || GROUP == "CPU" || GROUP == "ALL" || GROUP == "GPU"
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
