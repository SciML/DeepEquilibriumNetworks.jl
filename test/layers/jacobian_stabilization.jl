using DeepEquilibriumNetworks, Lux, NNlib, Random, Statistics, Zygote
using Test

include("../test_utils.jl")

function test_jacobian_trace_estimation()
    rng = get_prng(0)
    model = Parallel(+, get_dense_layer(3 => 3), get_dense_layer(3 => 3))
    ps, st = Lux.setup(rng, model)

    z = randn(rng, Float32, 3, 2)
    x = randn(rng, Float32, 3, 2)

    reverse_mode_estimate = estimate_jacobian_trace(Val(:reverse),
        model,
        ps,
        st,
        z,
        x,
        Lux.replicate(rng))
    run_JET_tests(estimate_jacobian_trace,
        Val(:reverse),
        model,
        ps,
        st,
        z,
        x,
        Lux.replicate(rng))

    finite_diff_estimate = estimate_jacobian_trace(Val(:finite_diff),
        model,
        ps,
        st,
        z,
        x,
        Lux.replicate(rng))
    run_JET_tests(estimate_jacobian_trace,
        Val(:finite_diff),
        model,
        ps,
        st,
        z,
        x,
        Lux.replicate(rng))

    @test isapprox(reverse_mode_estimate, finite_diff_estimate; atol=1e-1, rtol=1e-1)

    model = Parallel(+, Conv((1, 1), 3 => 3, relu), Conv((1, 1), 3 => 3))
    ps, st = Lux.setup(rng, model)

    z = randn(rng, Float32, 5, 5, 3, 2)
    x = randn(rng, Float32, 5, 5, 3, 2)

    reverse_mode_estimate = estimate_jacobian_trace(Val(:reverse),
        model,
        ps,
        st,
        z,
        x,
        Lux.replicate(rng))
    run_JET_tests(estimate_jacobian_trace,
        Val(:reverse),
        model,
        ps,
        st,
        z,
        x,
        Lux.replicate(rng))

    finite_diff_estimate = estimate_jacobian_trace(Val(:finite_diff),
        model,
        ps,
        st,
        z,
        x,
        Lux.replicate(rng))
    run_JET_tests(estimate_jacobian_trace,
        Val(:finite_diff),
        model,
        ps,
        st,
        z,
        x,
        Lux.replicate(rng))

    @test isapprox(reverse_mode_estimate, finite_diff_estimate; atol=1e-1, rtol=1e-1)

    return nothing
end

function test_jacobian_trace_estimation_gradient()
    rng = get_prng(0)
    model = Parallel(+, get_dense_layer(3 => 3), get_dense_layer(3 => 3))
    ps, st = Lux.setup(rng, model)

    z = randn(rng, Float32, 3, 2)
    x = randn(rng, Float32, 3, 2)

    gs = Zygote.gradient(ps -> estimate_jacobian_trace(Val(:finite_diff),
            model,
            ps,
            st,
            z,
            x,
            Lux.replicate(rng)),
        ps)[1]

    @test is_finite_gradient(gs)

    model = Parallel(+, Conv((1, 1), 3 => 3, relu), Conv((1, 1), 3 => 3))
    ps, st = Lux.setup(rng, model)

    z = randn(rng, Float32, 5, 5, 3, 2)
    x = randn(rng, Float32, 5, 5, 3, 2)

    gs = Zygote.gradient(ps -> estimate_jacobian_trace(Val(:finite_diff),
            model,
            ps,
            st,
            z,
            x,
            Lux.replicate(rng)),
        ps)[1]

    @test is_finite_gradient(gs)

    return nothing
end

@testset "Jacobian Trace Extimation" begin
    test_jacobian_trace_estimation()
end
@testset "Jacobian Trace Extimation: Gradient" begin
    test_jacobian_trace_estimation_gradient()
end
