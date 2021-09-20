using FastDEQ
using Flux
using OrdinaryDiffEq
using SteadyStateDiffEq
using CUDA
using Plots

CUDA.allowscalar(false)

struct Model{B1,B2,F}
    branch_1::B1
    branch_2::B2
    final::F
end

Flux.@functor Model

(mlp::Model)(x, y) = mlp.final(gelu.(mlp.branch_1(x) .+ mlp.branch_2(y)))

batch_size = 64
x_data = rand(Float32, 1, 256) .* 2 .- 1 |> gpu
y_data = x_data .^ 2 |> gpu

x_data_partition = (x_data[:, (i - 1) * batch_size + 1:i * batch_size] for i in 1:size(x_data, 2) ÷ batch_size)
y_data_partition = (y_data[:, (i - 1) * batch_size + 1:i * batch_size] for i in 1:size(y_data, 2) ÷ batch_size)

model = DeepEquilibriumNetwork(
    Model(Dense(1, 2), Dense(1, 2), Dense(2, 1)),
    DynamicSS(Tsit5(); abstol = 1f-2, reltol = 1f-2)
) |> gpu

function mse_loss(x, y)
    loss = mean(abs2, model(x) .- y)
    @show loss
    return loss
end

ps = Flux.params(model)
@show mse_loss(x_data, y_data)
Flux.@epochs 1000 Flux.train!(mse_loss, ps, zip(x_data_partition, y_data_partition), ADAMW(0.001))
@show mse_loss(x_data, y_data)

idx = sortperm(vec(x_data) |> cpu)

plot(vec(x_data)[idx] |> cpu, vec(y_data)[idx] |> cpu, label = "Ground Truth", legend = :topleft, lw = 2)
plot!(vec(x_data)[idx] |> cpu, vec(model(x_data))[idx] |> cpu, label = "Learned Model", lw = 2)