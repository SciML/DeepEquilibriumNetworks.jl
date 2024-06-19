@testsetup module SharedTestSetup

using DeepEquilibriumNetworks, Functors, Lux, Random, StableRNGs, Zygote, ForwardDiff
import LuxTestUtils: @jet
using LuxCUDA

CUDA.allowscalar(false)

__nameof(::X) where {X} = nameof(X)

__get_prng(seed::Int) = StableRNG(seed)

__is_finite_gradient(x::AbstractArray) = all(isfinite, x)

function __is_finite_gradient(gs::NamedTuple)
    gradient_is_finite = Ref(true)
    function __is_gradient_finite(x)
        !isnothing(x) && !all(isfinite, x) && (gradient_is_finite[] = false)
        return x
    end
    fmap(__is_gradient_finite, gs)
    return gradient_is_finite[]
end

function __get_dense_layer(args...; kwargs...)
    init_weight(rng::AbstractRNG, dims...) = randn(rng, Float32, dims) .* 0.001f0
    return Dense(args...; init_weight, use_bias=false, kwargs...)
end

function __get_conv_layer(args...; kwargs...)
    init_weight(rng::AbstractRNG, dims...) = randn(rng, Float32, dims) .* 0.001f0
    return Conv(args...; init_weight, use_bias=false, kwargs...)
end

const GROUP = get(ENV, "GROUP", "All")

cpu_testing() = GROUP == "All" || GROUP == "CPU"
cuda_testing() = LuxCUDA.functional() && (GROUP == "All" || GROUP == "CUDA")

const MODES = begin
    cpu_mode = ("CPU", Array, LuxCPUDevice(), false)
    cuda_mode = ("CUDA", CuArray, LuxCUDADevice(), true)

    modes = []
    cpu_testing() && push!(modes, cpu_mode)
    cuda_testing() && push!(modes, cuda_mode)

    modes
end

export Lux, LuxCore, LuxLib
export MODES, __get_dense_layer, __get_conv_layer, __is_finite_gradient, __get_prng,
       __nameof, @jet

end
