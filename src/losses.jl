Base.@kwdef struct SupervisedLossContainer{L,T}
    loss_function::L
    λ::T = 0.0f0
    λⱼ::T = 0.0f0
    λᵣ::T = 0.0f0
end

function (lc::SupervisedLossContainer)(soln::DeepEquilibriumSolution)
    return lc.λ * mean(abs, soln.u₀ .- soln.z_star) + lc.λⱼ * soln.jacobian_loss + lc.λᵣ * norm(soln.residual)
end

function (lc::SupervisedLossContainer)(soln::DeepEquilibriumSolution{T}) where {T<:Tuple}
    return lc.λ * mapreduce((x, y) -> mean(abs, x .- y), +, soln.u₀, soln.z_star) +
           lc.λⱼ * soln.jacobian_loss +
           lc.λᵣ * sum(norm, soln.residual)
end

function (lc::SupervisedLossContainer)(model::Union{DeepEquilibriumNetwork,SkipDeepEquilibriumNetwork,DEQChain}, x, y;
                                       kwargs...)
    ŷ, soln = model(x; kwargs...)
    return lc.loss_function(ŷ, y) + lc(soln)
end

function (lc::SupervisedLossContainer)(model::Union{MultiScaleDeepEquilibriumNetwork,
                                                    MultiScaleSkipDeepEquilibriumNetwork}, x, ys::Tuple; kwargs...)
    yŝ, soln = model(x; kwargs...)
    return mapreduce(lc.loss_function, +, ys, yŝ) + lc(soln)
end

function (lc::SupervisedLossContainer)(model::Union{MultiScaleDeepEquilibriumNetwork,
                                                    MultiScaleSkipDeepEquilibriumNetwork}, x, y; kwargs...)
    yŝ, soln = model(x; kwargs...)
    return sum(Base.Fix2(lc.loss_function, y), yŝ) + lc(soln)
end

function (lc::SupervisedLossContainer)(model::DataParallelFluxModel, args...; kwargs...)
    return lc(model.model, args...; kwargs...)
end

## FIXME: It is not recommended to use WidthStackedDEQ
# function (lc::SupervisedLossContainer)(model::WidthStackedDEQ{true}, x, y; kwargs...)
#     ŷ, guess_pairs = model(x; kwargs...)
#     l1 = lc.loss_function(ŷ, y)
#     l2 = 0
#     for (y, ŷ) in guess_pairs
#         l2 += mean(abs, y .- ŷ)
#     end
#     return l1 + lc.λ * l2
# end

# Default fallback
function (lc::SupervisedLossContainer)(model, x, y; kwargs...)
    return lc.loss_function(model(x; kwargs...), y)
end
