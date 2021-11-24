Base.@kwdef struct SupervisedLossContainer{L,T}
    loss_function::L
    λ::T = 1.0f0
end

(lc::SupervisedLossContainer)(
    model::Union{DEQChain{Val(1)},DEQChain{Val(3)},WidthStackedDEQ{false}},
    x,
    y;
    kwargs...
) = lc.loss_function(model(x), y)

function (lc::SupervisedLossContainer)(model::DEQChain{Val(2)}, x, y; kwargs...)
    ŷ, (ẑ, z) = model(x)
    return lc.loss_function(ŷ, y) + lc.λ * mean(abs, ẑ .- z)
end

function (lc::SupervisedLossContainer)(model::DEQChain{Val(4)}, x, y; kwargs...)
    ŷ, guess_pairs = model(x)
    l1 = lc.loss_function(ŷ, y)
    l2 = 0
    for i = 1:length(guess_pairs[1])
        l2 += mean(abs, guess_pairs[1][i] .- guess_pairs[2][i])
    end
    return l1 + sum(lc.λ .* l2)
end

function (lc::SupervisedLossContainer)(model::WidthStackedDEQ{true}, x, y; kwargs...)
    ŷ, guess_pairs = model(x)
    l1 = lc.loss_function(ŷ, y)
    l2 = 0
    for (y, ŷ) in guess_pairs
        l2 += mean(abs, y .- ŷ)
    end
    return l1 + lc.λ * l2
end
