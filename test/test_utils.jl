using DeepEquilibriumNetworks, Functors, Lux, Random, StableRNGs, Zygote
import LuxTestUtils: @jet

__nameof(::X) where {X} = nameof(X)

__get_prng(seed::Int) = StableRNG(seed)

# is_finite_gradient(x::AbstractArray) = all(isfinite, x)

# function is_finite_gradient(gs::NamedTuple)
#     gradient_is_finite = [true]
#     function _is_gradient_finite(x)
#         if !isnothing(x) && !all(isfinite, x)
#             gradient_is_finite[1] = false
#         end
#         return x
#     end
#     Functors.fmap(_is_gradient_finite, gs)
#     return gradient_is_finite[1]
# end

function __get_dense_layer(args...; kwargs...)
    init_weight(rng::AbstractRNG, dims...) = randn(rng, Float32, dims) .* 0.001f0
    return Dense(args...; init_weight, use_bias=false, kwargs...)
end

function __get_conv_layer(args...; kwargs...)
    init_weight(rng::AbstractRNG, dims...) = randn(rng, Float32, dims) .* 0.001f0
    return Conv(args...; init_weight, use_bias=false, kwargs...)
end
