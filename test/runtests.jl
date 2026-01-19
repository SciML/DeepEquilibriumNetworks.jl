using Pkg
using SafeTestsets, Test

const GROUP = uppercase(get(ENV, "GROUP", "CPU"))

@info "Running tests for GROUP: $GROUP"

@time begin
    if GROUP == "CPU" || GROUP == "ALL"
        @time @safetestset "Utils Tests" include("utils_tests.jl")
        @time @safetestset "Layers Tests" include("layers_tests.jl")
    end

    if GROUP == "QA"
        @time @safetestset "Quality Assurance Tests" include("qa_tests.jl")
    end
end
