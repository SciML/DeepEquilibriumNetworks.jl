using DeepEquilibriumNetworks, Functors, Lux, Random, StableRNGs, Zygote, ForwardDiff, Test
using LuxTestUtils
using MLDataDevices, GPUArraysCore

const DEQs = DeepEquilibriumNetworks

LuxTestUtils.jet_target_modules!(["DeepEquilibriumNetworks", "Lux", "LuxLib"])

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", get(ENV, "GROUP", "cpu")))

if BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda"
    using LuxCUDA
end

GPUArraysCore.allowscalar(false)

cpu_testing() = BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
function cuda_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") &&
           MLDataDevices.functional(CUDADevice)
end

const MODES = begin
    modes = []
    cpu_testing() && push!(modes, ("cpu", Array, CPUDevice(), false))
    cuda_testing() && push!(modes, ("cuda", CuArray, CUDADevice(), true))
    modes
end

is_finite_gradient(x::AbstractArray) = all(isfinite, x)
is_finite_gradient(::Nothing) = true
is_finite_gradient(gs) = all(is_finite_gradient, fleaves(gs))

function dense_layer(args...; kwargs...)
    init_weight(rng::AbstractRNG, dims...) = randn(rng, Float32, dims) .* 0.001f0
    return Dense(args...; init_weight, use_bias=false, kwargs...)
end

function conv_layer(args...; kwargs...)
    init_weight(rng::AbstractRNG, dims...) = randn(rng, Float32, dims) .* 0.001f0
    return Conv(args...; init_weight, use_bias=false, kwargs...)
end
