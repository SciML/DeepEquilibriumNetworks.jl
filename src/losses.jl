Base.@kwdef struct SupervisedLossContainer{L,T}
    loss_function::L
    λ::T = 1.0f0
end

function (lc::SupervisedLossContainer)(model::DEQChain{false}, x, y; kwargs...)
    l1 = lc.loss_function(model(x), y)
    return l1
end

function (lc::SupervisedLossContainer)(model::DEQChain{true}, x, y; kwargs...)
    ŷ, (ẑ, z) = model(x)
    l1 = lc.loss_function(ŷ, y)
    l2 = mean(abs2, ẑ .- z)
    return l1 + lc.λ * l2
end
