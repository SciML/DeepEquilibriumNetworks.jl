Base.@kwdef struct SupervisedLossContainer{L,T<:Real}
    loss_function::L
    λ::T = 1.0f0
end

(lc::SupervisedLossContainer)(model::DEQChain{false}, x, y) =
    lc.loss_function(model(x), y)

function (lc::SupervisedLossContainer)(model::DEQChain{true}, x, y)
    ŷ, (ẑ, z) = model(x)
    l1 = lc.loss_function(ŷ, y)
    l2 = mean(abs2, ẑ .- z)
    return l1 + lc.λ * l2
end