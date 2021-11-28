Base.@kwdef struct SupervisedLossContainer{L,T}
    loss_function::L
    λ::T = 1.0f0
end

(lc::SupervisedLossContainer)(model::DEQChain{Val(1)}, x, y; kwargs...) =
    lc.loss_function(model(x), y)

function (lc::SupervisedLossContainer)(model::DEQChain{Val(2)}, x, y; kwargs...)
    ŷ, (ẑ, z) = model(x)
    return lc.loss_function(ŷ, y) + lc.λ * mean(abs, ẑ .- z)
end

(lc::SupervisedLossContainer)(model::DEQChain{Val(3)}, x, y; kwargs...) =
    lc.loss_function(model(x), y)

function (lc::SupervisedLossContainer)(model::DEQChain{Val(4)}, x, y; kwargs...)
    ŷ, guess_pairs = model(x)
    l1 = lc.loss_function(ŷ, y)
    l2 = 0
    for i = 1:length(guess_pairs[1])
        l2 += mean(abs, guess_pairs[1][i] .- guess_pairs[2][i])
    end
    return l1 + sum(lc.λ .* l2)
end

(lc::SupervisedLossContainer)(model::WidthStackedDEQ{false}, x, y; kwargs...) =
    lc.loss_function(model(x), y)

(lc::SupervisedLossContainer)(model::DeepEquilibriumNetwork, x, y; kwargs...) =
    lc.loss_function(model(x), y)

function (lc::SupervisedLossContainer)(
    model::SkipDeepEquilibriumNetwork,
    x,
    y;
    kwargs...,
)
    ŷ, ẑ = model(x)
    return lc.loss_function(ŷ, y) + lc.λ * mean(abs, ẑ .- ŷ)
end

function (lc::SupervisedLossContainer)(
    model::WidthStackedDEQ{true},
    x,
    y;
    kwargs...,
)
    ŷ, guess_pairs = model(x)
    l1 = lc.loss_function(ŷ, y)
    l2 = 0
    for (y, ŷ) in guess_pairs
        l2 += mean(abs, y .- ŷ)
    end
    return l1 + lc.λ * l2
end

(lc::SupervisedLossContainer)(
    model::DataParallelFluxModel,
    args...;
    kwargs...,
) = lc(model.model, args...; kwargs...)

(lc::SupervisedLossContainer)(
    model::MultiScaleDeepEquilibriumNetwork,
    x,
    ys::Tuple;
    kwargs...,
) = sum(ŷy -> lc.loss_function(ŷy[1], ŷy[2]), zip(model(x), ys))

(lc::SupervisedLossContainer)(
    model::MultiScaleDeepEquilibriumNetwork,
    x,
    y;
    kwargs...,
) = sum(ŷ -> lc.loss_function(ŷ, y), model(x))

function (lc::SupervisedLossContainer)(
    model::MultiScaleSkipDeepEquilibriumNetwork,
    x,
    ys::Tuple;
    kwargs...,
)
    ŷs, guesses = model(x)
    l1 = sum(ŷy -> lc.loss_function(ŷy[1], ŷy[2]), zip(ŷs, ys))
    l2 = sum(ŷy -> mean(abs, ŷy[1] .- ŷy[2]), zip(ŷs, guesses))
    return l1 + lc.λ * l2
end