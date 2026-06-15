using SciMLTesting
using SafeTestsets

# DEQ's GROUP semantics are backend/capability-based, not folder-partitioned: the
# GPU group runs the *same* Core test files (utils_tests.jl + layers_tests.jl)
# with shared_testsetup.jl's backend switched to CUDA via BACKEND_GROUP, so it
# cannot be expressed as a separate folder of files. Hence explicit-args run_tests.
function core_body()
    @safetestset "Utils Tests" include("utils_tests.jl")
    return @safetestset "Layers Tests" include("layers_tests.jl")
end

run_tests(;
    core = core_body,
    groups = Dict(
        # GPU is the self-hosted CUDA runner lane: the same Core suite with the
        # backend switched to CUDA.
        "GPU" => () -> begin
            ENV["BACKEND_GROUP"] = "CUDA"
            core_body()
        end,
    ),
    qa = (; env = joinpath(@__DIR__, "qa"), body = joinpath(@__DIR__, "qa", "qa.jl")),
    # Curated "All": run only Core. GPU (self-hosted CUDA lane) and QA stay
    # selectable by name but out of the aggregate.
    all = ["Core"],
)
