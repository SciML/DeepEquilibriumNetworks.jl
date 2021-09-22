using FastDEQ
using Flux
using OrdinaryDiffEq
using SteadyStateDiffEq
using CUDA
using Plots
using Random

Random.seed!(1)
CUDA.allowscalar(false)

struct Model{B1,B2,F}
    branch_1::B1
    branch_2::B2
    final::F
end

Flux.@functor Model

(mlp::Model)(x, y) = mlp.final(tanh.(mlp.branch_1(x) .+ mlp.branch_2(y)))

batch_size = 64
x_data = rand(Float32, 1, 256) .* 2 .- 1 |> gpu
y_data = x_data .^ 2 |> gpu

x_data_partition = (
    x_data[:, (i-1)*batch_size+1:i*batch_size] for
    i = 1:size(x_data, 2)÷batch_size
)
y_data_partition = (
    y_data[:, (i-1)*batch_size+1:i*batch_size] for
    i = 1:size(y_data, 2)÷batch_size
)

function register_nfe_counts(deq, buffer)
    function callback()
        push!(buffer, deq.stats.nfe)
        deq.stats.nfe = 0
    end
    return callback
end

# Standard Model

model =
    DeepEquilibriumNetwork(
        Model(Dense(1, 100), Dense(1, 100), Dense(100, 1)),
        DynamicSS(Tsit5(); abstol = 1f-2, reltol = 1f-2),
    ) |> gpu

function mse_loss(x, y)
    loss = mean(abs2, model(x) .- y)
    @show loss
    return loss
end

nfe_counts_1 = Int64[]

ps = Flux.params(model)
@show mse_loss(x_data, y_data)
Flux.@epochs 1000 Flux.train!(
    mse_loss,
    ps,
    zip(x_data_partition, y_data_partition),
    ADAMW(0.001),
    cb = register_nfe_counts(model, nfe_counts_1),
)
@show mse_loss(x_data, y_data)

idx = sortperm(vec(x_data) |> cpu);

plot(
    vec(x_data)[idx] |> cpu,
    vec(y_data)[idx] |> cpu,
    label = "Ground Truth",
    legend = :topleft,
    lw = 2,
)
plot!(
    vec(x_data)[idx] |> cpu,
    vec(model(x_data))[idx] |> cpu,
    label = "Learned Model",
    lw = 2,
)
savefig("experiments/figures/deq_polynomial_fit.png")

plot(nfe_counts_1, label = "Vanilla DEQ NFE")
savefig("experiments/figures/deq_polynomial_fit_nfe.png")


# FastDEQ Model

model2 =
    SkipDeepEquilibriumNetwork(
        Model(Dense(1, 100), Dense(1, 100), Dense(100, 1)),
        Chain(Dense(1, 16, relu), Dense(16, 1)),
        DynamicSS(Tsit5(); abstol = 1f-2, reltol = 1f-2),
    ) |> gpu

function mse_loss2(x, y)
    ŷ, s = model2(x)
    loss = mean(abs2, ŷ .- y)
    @show loss
    return loss + 0.01f0 * mean(abs2, ŷ .- s)
end

nfe_counts_2 = Int64[]

ps = Flux.params(model2)
@show mse_loss2(x_data, y_data)
Flux.@epochs 1000 Flux.train!(
    mse_loss2,
    ps,
    zip(x_data_partition, y_data_partition),
    ADAMW(0.001),
    cb = register_nfe_counts(model2, nfe_counts_2)
)
@show mse_loss2(x_data, y_data)

idx = sortperm(vec(x_data) |> cpu);

plot(
    vec(x_data)[idx] |> cpu,
    vec(y_data)[idx] |> cpu,
    label = "Ground Truth",
    legend = :topleft,
    lw = 2,
)
plot!(
    vec(x_data)[idx] |> cpu,
    vec(model(x_data))[idx] |> cpu,
    label = "Learned Model",
    lw = 2,
)
savefig("experiments/figures/skipdeq_polynomial_fit.png")

plot(nfe_counts_2, label = "SkipDEQ NFE")
savefig("experiments/figures/skipdeq_polynomial_fit_nfe.png")
