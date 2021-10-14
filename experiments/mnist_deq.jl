using DataLoaders: DataLoader
using MLDataPattern: splitobs, shuffleobs
using ProgressBars
using Flux
using FastDEQ
using OrdinaryDiffEq
using SteadyStateDiffEq

# Resnet Layer
struct ResNetLayer{C1,C2,N1,N2,N3}
    conv1::C1
    conv2::C2
    norm1::N1
    norm2::N2
    norm3::N3
end

Flux.@functor ResNetLayer

function ResNetLayer(
    n_channels::Int,
    n_inner_channels::Int,
    kernel_size::Tuple{Int,Int} = (3, 3),
    num_groups::Int = 8,
)
    conv1 = Conv(
        kernel_size,
        n_channels => n_inner_channels,
        relu;
        pad = kernel_size .÷ 2,
        bias = false,
        init = (dims...) -> randn(Float32, dims...) .* 0.01f0,
    )
    conv2 = Conv(
        kernel_size,
        n_inner_channels => n_channels;
        pad = kernel_size .÷ 2,
        bias = false,
        init = (dims...) -> randn(Float32, dims...) .* 0.01f0,
    )
    norm1 = GroupNorm(n_inner_channels, num_groups, affine = false)
    norm2 = GroupNorm(n_channels, num_groups, affine = false)
    norm3 = GroupNorm(n_channels, num_groups, affine = false)

    return ResNetLayer(conv1, conv2, norm1, norm2, norm3)
end

(rl::ResNetLayer)(z, x) =
    rl.norm3(relu.(z .+ rl.norm2(x .+ rl.conv2(rl.norm1(rl.conv1(z))))))


xs, ys = (
    # convert each image into h*w*1 array of floats 
    [Float32.(reshape(img, 28, 28, 1)) for img in Flux.Data.MNIST.images()],
    # one-hot encode the labels
    [Float32.(Flux.onehot(y, 0:9)) for y in Flux.Data.MNIST.labels()],
)

# split into training and validation sets
traindata, valdata = splitobs((xs, ys), at = 0.9)

# create iterators
valiter = DataLoader(valdata, 512, buffered = false);

# Set device
dev = gpu

# Final Model
model =
    Chain(
        Conv((3, 3), 1 => 48, relu; bias = true, pad = 1),  # 28 x 28 x 48
        BatchNorm(48, affine = true),
        DeepEquilibriumNetwork(
            ResNetLayer(48, 64) |> dev,
            DynamicSS(Tsit5(); abstol = 1.0f-1),
            maxiters = 40,
            sensealg = SteadyStateAdjoint(
                autodiff = false,
                autojacvec = ZygoteVJP(),
                linsolve = LinSolveKrylovJL(rtol = T(0.001), atol = T(0.001)),
            ),
        ),
        BatchNorm(48, affine = true),
        MeanPool((8, 8)),  # 3 x 3 x 48
        Flux.flatten,
        Dense(3 * 3 * 48, 10),
    ) |> dev

# Utilities
lossfn = Flux.Losses.logitcrossentropy
opt = Flux.ADAM()

function loss_and_accuracy(model, dataloader, device)
    matches, total_loss, total_datasize = 0, 0, 0
    iter = ProgressBar(dataloader)
    for (x, y) in iter
        x = x |> device
        y = y |> device

        ŷ = model(x)
        total_loss += lossfn(ŷ, y) * size(x, ndims(x))
        matches += sum(argmax.(eachcol(ŷ)) .== Flux.onecold(y |> cpu))
        total_datasize += size(x, ndims(x))

        set_description(
            iter,
            "Val Loss: $(total_loss / total_datasize) | Val Accuracy: $(matches / total_datasize)",
        )
    end
    return total_loss / total_datasize, matches / total_datasize
end

# Training
ps = Flux.params(model)
for epoch = 1:100
    epoch_loss = 0
    epoch_count = 0
    trainiter = DataLoader(shuffleobs(traindata), 512, buffered = false);
    iter = ProgressBar(trainiter)
    for (x, y) in iter
        x = x |> dev
        y = y |> dev

        loss, back = Flux.pullback(() -> lossfn(model(x), y), ps)
        gs = back(one(loss))
        Flux.Optimise.update!(opt, ps, gs)

        epoch_loss += loss * size(x, ndims(x))
        epoch_count += size(x, ndims(x))

        set_description(iter, "Training Loss: $(epoch_loss / epoch_count)")
    end
    @show loss_and_accuracy(model, valiter, dev)
end