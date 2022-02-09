"""
    SupervisedLossContainer(loss_function)
    SupervisedLossContainer(loss_function, λ, λⱼ)

A container class for supervised loss functions.
"""
Base.@kwdef struct SupervisedLossContainer{L,T}
    loss_function::L
    λ::T = 0.0f0
    λⱼ::T = 0.0f0
end

function (lc::SupervisedLossContainer)(soln::DeepEquilibriumSolution)
    return lc.λ * mean(abs, soln.u₀ .- soln.z_star) + lc.λⱼ * soln.jacobian_loss
end

function (lc::SupervisedLossContainer)(soln::DeepEquilibriumSolution{T}) where {T<:Tuple}
    return lc.λ * mapreduce((x, y) -> mean(abs, x .- y), +, soln.u₀, soln.z_star) +
           lc.λⱼ * soln.jacobian_loss
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

# Default fallback
function (lc::SupervisedLossContainer)(model, x, y; kwargs...)
    return lc.loss_function(model(x; kwargs...), y)
end
