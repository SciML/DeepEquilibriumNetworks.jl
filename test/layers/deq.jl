using DeepEquilibriumNetworks, DiffEqBase, Lux, OrdinaryDiffEq, SimpleNonlinearSolve
using Test

include("../test_utils.jl")

function DEFAULT_DEQ_SOLVERS()
    termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.RelSafe;
        abstol=0.01f0, reltol=0.01f0)

    return (ContinuousDEQSolver(VCABM3(); abstol=0.01f0, reltol=0.01f0),
        DiscreteDEQSolver(LBroyden(; batched=true, termination_condition)))
end

function test_deep_equilibrium_network()
    rng = get_prng(0)

    for solver in DEFAULT_DEQ_SOLVERS()
        model = DeepEquilibriumNetwork(Parallel(+,
                get_dense_layer(2, 2; use_bias=false),
                get_dense_layer(2, 2; use_bias=false)),
            solver; verbose=false, save_everystep=true)

        ps, st = Lux.setup(rng, model)

        @test st.solution === nothing

        x = randn(rng, Float32, 2, 1)

        z, st = model(x, ps, st)

        @test all(isfinite, z)
        @test size(z) == size(x)
        @test st.solution isa DeepEquilibriumSolution

        ps, st = Lux.setup(rng, model)
        st = Lux.update_state(st, :fixed_depth, Val(10))

        @test st.solution === nothing

        z, st = model(x, ps, st)

        @test all(isfinite, z)
        @test size(z) == size(x)
        @test st.solution isa DeepEquilibriumSolution
        @test st.solution.nfe == 10
    end

    return nothing
end

function test_skip_deep_equilibrium_network()
    rng = get_prng(0)

    for solver in DEFAULT_DEQ_SOLVERS()
        model = SkipDeepEquilibriumNetwork(Parallel(+,
                get_dense_layer(2, 2; use_bias=false),
                get_dense_layer(2, 2; use_bias=false)),
            get_dense_layer(2, 2), solver; verbose=false, save_everystep=true)

        ps, st = Lux.setup(rng, model)

        @test st.solution === nothing

        x = randn(rng, Float32, 2, 1)

        z, st = model(x, ps, st)

        @test all(isfinite, z)
        @test size(z) == size(x)
        @test st.solution isa DeepEquilibriumSolution

        ps, st = Lux.setup(rng, model)
        st = Lux.update_state(st, :fixed_depth, Val(10))

        @test st.solution === nothing

        z, st = model(x, ps, st)

        @test all(isfinite, z)
        @test size(z) == size(x)
        @test st.solution isa DeepEquilibriumSolution
        @test st.solution.nfe == 10
    end

    return nothing
end

function test_skip_deep_equilibrium_network_v2()
    rng = get_prng(0)

    for solver in DEFAULT_DEQ_SOLVERS()
        model = SkipDeepEquilibriumNetwork(Parallel(+,
                get_dense_layer(2, 2; use_bias=false),
                get_dense_layer(2, 2; use_bias=false)),
            nothing, solver; verbose=false, save_everystep=true)

        ps, st = Lux.setup(rng, model)

        @test st.solution === nothing

        x = randn(rng, Float32, 2, 1)

        z, st = model(x, ps, st)

        @test all(isfinite, z)
        @test size(z) == size(x)
        @test st.solution isa DeepEquilibriumSolution

        ps, st = Lux.setup(rng, model)
        st = Lux.update_state(st, :fixed_depth, Val(10))

        @test st.solution === nothing

        z, st = model(x, ps, st)

        @test all(isfinite, z)
        @test size(z) == size(x)
        @test st.solution isa DeepEquilibriumSolution
        @test st.solution.nfe == 10
    end

    return nothing
end

@testset "DeepEquilibriumNetwork" begin
    test_deep_equilibrium_network()
end
@testset "SkipDeepEquilibriumNetwork" begin
    test_skip_deep_equilibrium_network()
end
@testset "SkipRegDeepEquilibriumNetwork" begin
    test_skip_deep_equilibrium_network_v2()
end
