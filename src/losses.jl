Base.@kwdef struct SupervisedLossContainer{L,T}
    loss_function::L
    λ::T = 1.0f0
end

function (lc::SupervisedLossContainer)(model::Union{DEQChain{Val(1)},DEQChain{Val(3)}}, x, y; kwargs...)
    l1 = lc.loss_function(model(x), y)
    return l1
end

function (lc::SupervisedLossContainer)(model::DEQChain{Val(2)}, x, y; kwargs...)
    ŷ, (ẑ, z) = model(x)
    l1 = lc.loss_function(ŷ, y)
    l2 = mean(abs2, ẑ .- z)
    return l1 + lc.λ * l2
end

function (lc::SupervisedLossContainer)(model::DEQChain{Val(4)}, x, y; kwargs...)
    ŷ, guess_pairs = model(x)
    l1 = lc.loss_function(ŷ, y)
    l2 = map(z -> mean(abs2, z[2] .- z[1]), guess_pairs)
    return l1 + sum(lc.λ .* l2)
end
