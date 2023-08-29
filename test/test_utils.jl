# TODO: Migrate to LuxTestUtils.jl
using DeepEquilibriumNetworks, Functors, Lux, Random

global test_call(args...; kwargs...) = nothing
global test_opt(args...; kwargs...) = nothing

try
    import JET
    global test_call(args...; kwargs...) = JET.test_call(args...; kwargs...)
    global test_opt(args...; kwargs...) = JET.test_opt(args...; kwargs...)
catch
    @warn "JET not precompiling. All JET tests will be skipped." maxlog=1
end

function get_prng(seed::Int)
    @static if VERSION >= v"1.7"
        rng = Xoshiro()
        Random.seed!(rng, seed)
        return rng
    else
        rng = MersenneTwister()
        Random.seed!(rng, seed)
        return rng
    end
end

is_finite_gradient(x::AbstractArray) = all(isfinite, x)

function is_finite_gradient(gs::NamedTuple)
    gradient_is_finite = [true]
    function _is_gradient_finite(x)
        if !isnothing(x) && !all(isfinite, x)
            gradient_is_finite[1] = false
        end
        return x
    end
    Functors.fmap(_is_gradient_finite, gs)
    return gradient_is_finite[1]
end

function run_JET_tests(f, args...; call_broken=false, opt_broken=false, kwargs...)
    @static if VERSION >= v"1.7"
        test_call(f, typeof.(args); broken=call_broken, target_modules=(DEQs,))
        test_opt(f, typeof.(args); broken=opt_broken, target_modules=(DEQs,))
    end
end

function get_dense_layer(args...; kwargs...)
    function init_weight(rng::Random.AbstractRNG, out_dims, in_dims)
        return randn(rng, Float32, (out_dims, in_dims)) .* 0.001f0
    end
    return Dense(args...; init_weight, kwargs...)
end
