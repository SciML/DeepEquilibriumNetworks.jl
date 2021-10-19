Base.@kwdef struct SupervisedLossContainer{L,T<:Real}
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

Base.@kwdef mutable struct ScheduledSupervisedLossContainer{L,T<:Real}
    loss_function::L
    max_counter::Int
    λ_start::T = 0.0f0
    λ_end::T = 1.0f0
    counter::Int = 0
end

function (lc::ScheduledSupervisedLossContainer)(
    model::DEQChain{false},
    x,
    y;
    update_counter::Bool = true,
    kwargs...,
)
    Zygote.@ignore begin
        update_counter && (lc.counter += 1)
    end
    l1 = lc.loss_function(model(x), y)
    return l1
end

function (lc::ScheduledSupervisedLossContainer)(
    model::DEQChain{true},
    x,
    y;
    update_counter::Bool = true,
    kwargs...,
)
    Zygote.@ignore begin
        update_counter && (lc.counter += 1)
    end
    ŷ, (ẑ, z) = model(x)
    l1 = lc.loss_function(ŷ, y)
    l2 = mean(abs2, ẑ .- z)
    λ = (lc.λ_end - lc.λ_start) * min(lc.counter / lc.max_counter, 1)
    return l1 + λ * l2
end
