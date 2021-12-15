Base.@kwdef struct SupervisedLossContainer{L,T}
    loss_function::L
    λ::T = 1.0f0
end

function (lc::SupervisedLossContainer)(model::DEQChain{Val(2)}, x, y; kwargs...)
    ŷ, (ẑ, z) = model(x; kwargs...)
    return lc.loss_function(ŷ, y) + lc.λ * mean(abs, ẑ .- z)
end

function (lc::SupervisedLossContainer)(model::DEQChain{Val(4)}, x, y; kwargs...)
    ŷ, guess_pairs = model(x; kwargs...)
    l1 = lc.loss_function(ŷ, y)
    l2 = 0
    for i in 1:length(guess_pairs[1])
        l2 += mean(abs, guess_pairs[1][i] .- guess_pairs[2][i])
    end
    return l1 + lc.λ * l2
end

function (lc::SupervisedLossContainer)(model::WidthStackedDEQ{true}, x, y; kwargs...)
    ŷ, guess_pairs = model(x; kwargs...)
    l1 = lc.loss_function(ŷ, y)
    l2 = 0
    for (y, ŷ) in guess_pairs
        l2 += mean(abs, y .- ŷ)
    end
    return l1 + lc.λ * l2
end

function (lc::SupervisedLossContainer)(model::SkipDeepEquilibriumNetwork, x, y; kwargs...)
    ŷ, ẑ = model(x; kwargs...)
    return lc.loss_function(ŷ, y) + lc.λ * mean(abs, ẑ .- ŷ)
end

function (lc::SupervisedLossContainer)(model::MultiScaleDeepEquilibriumNetwork, x, ys::Tuple; kwargs...)
    return sum(ŷy -> lc.loss_function(ŷy[1], ŷy[2]), zip(model(x; kwargs...), ys))
end

function (lc::SupervisedLossContainer)(model::MultiScaleDeepEquilibriumNetwork, x, y; kwargs...)
    return sum(ŷ -> lc.loss_function(ŷ, y), model(x; kwargs...))
end

function (lc::SupervisedLossContainer)(model::MultiScaleSkipDeepEquilibriumNetwork, x, ys::Tuple; kwargs...)
    ŷs, guesses = model(x; kwargs...)
    l1 = sum(ŷy -> lc.loss_function(ŷy[1], ŷy[2]), zip(ŷs, ys))
    l2 = sum(ŷy -> mean(abs, ŷy[1] .- ŷy[2]), zip(ŷs, guesses))
    return l1 + lc.λ * l2
end

function (lc::SupervisedLossContainer)(model::DataParallelFluxModel, args...; kwargs...)
    return lc(model.model, args...; kwargs...)
end

# Default fallback
function (lc::SupervisedLossContainer)(model, x, y; kwargs...)
    return lc.loss_function(model(x; kwargs...), y)
end