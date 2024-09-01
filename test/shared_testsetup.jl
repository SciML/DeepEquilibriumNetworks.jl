@testsetup module SharedTestSetup

using DeepEquilibriumNetworks, Functors, Lux, Random, StableRNGs, Zygote, ForwardDiff
using LuxTestUtils
using MLDataDevices, GPUArraysCore

LuxTestUtils.jet_target_modules!(["DeepEquilibriumNetworks", "Lux", "LuxLib"])

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))

if BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda"
    using LuxCUDA
end

if BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu"
    using AMDGPU
end

GPUArraysCore.allowscalar(false)

cpu_testing() = BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
function cuda_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") &&
           MLDataDevices.functional(CUDADevice)
end
function amdgpu_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") &&
           MLDataDevices.functional(AMDGPUDevice)
end

const MODES = begin
    modes = []
    cpu_testing() && push!(modes, ("cpu", Array, CPUDevice(), false))
    cuda_testing() && push!(modes, ("cuda", CuArray, CUDADevice(), true))
    amdgpu_testing() && push!(modes, ("amdgpu", ROCArray, AMDGPUDevice(), true))
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

export Lux, LuxCore, LuxLib
export MODES, dense_layer, conv_layer, is_finite_gradient, StableRNG, @jet, test_gradients

end
