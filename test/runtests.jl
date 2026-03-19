using Pkg
using SafeTestsets, Test

const BACKEND_GROUP = uppercase(get(ENV, "BACKEND_GROUP", get(ENV, "GROUP", "CPU")))

@info "Running tests for BACKEND_GROUP: $BACKEND_GROUP"

@time begin
    if BACKEND_GROUP == "CPU" || BACKEND_GROUP == "ALL"
        @time @safetestset "Utils Tests" include("utils_tests.jl")
        @time @safetestset "Layers Tests" include("layers_tests.jl")
    end

    if BACKEND_GROUP == "CUDA" || BACKEND_GROUP == "ALL"
        @time @safetestset "CUDA Utils Tests" include("utils_tests.jl")
        @time @safetestset "CUDA Layers Tests" include("layers_tests.jl")
    end

    if BACKEND_GROUP == "QA"
        @time @safetestset "Quality Assurance Tests" include("qa_tests.jl")
    end
end
