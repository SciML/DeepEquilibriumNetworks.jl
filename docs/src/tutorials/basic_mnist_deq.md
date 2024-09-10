# Training a Simple MNIST Classifier using Deep Equilibrium Models

We will train a simple Deep Equilibrium Model on MNIST. First we load a few packages.

```@example basic_mnist_deq
using DeepEquilibriumNetworks, SciMLSensitivity, Lux, NonlinearSolve, OrdinaryDiffEq,
      Random, Optimisers, Zygote, LinearSolve, Dates, Printf, Setfield, OneHotArrays
using MLDatasets: MNIST
using MLUtils: DataLoader, splitobs
using LuxCUDA # For NVIDIA GPU support

CUDA.allowscalar(false)
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
```

Setup device functions from Lux. See
[GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management) for more details.

```@example basic_mnist_deq
const cdev = cpu_device()
const gdev = gpu_device()
```

We can now construct our dataloader. We are using only limited part of the data for
demonstration.

```@example basic_mnist_deq
function loadmnist(batchsize, train_split)
    N = 2500
    dataset = MNIST(; split=:train)
    imgs = dataset.features[:, :, 1:N]
    labels_raw = dataset.targets[1:N]

    # Process images into (H,W,C,BS) batches
    x_data = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)))
    y_data = onehotbatch(labels_raw, 0:9)
    (x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data); at=train_split)

    return (
        # Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize, shuffle=true),
        # Don't shuffle the test data
        DataLoader(collect.((x_test, y_test)); batchsize, shuffle=false))
end
```

Construct the Lux Neural Network containing a DEQ layer.

```@example basic_mnist_deq
function construct_model(solver; model_type::Symbol=:deq)
    down = Chain(Conv((3, 3), 1 => 64, gelu; stride=1), GroupNorm(64, 64),
        Conv((4, 4), 64 => 64; stride=2, pad=1))

    # The input layer of the DEQ
    deq_model = Chain(
        Parallel(+,
            Conv((3, 3), 64 => 64, tanh; stride=1, pad=SamePad(),
                init_weight=truncated_normal(std=0.01), use_bias=false),
            Conv((3, 3), 64 => 64, tanh; stride=1, pad=SamePad(),
                init_weight=truncated_normal(std=0.01), use_bias=false)),
        Conv((3, 3), 64 => 64, tanh; stride=1, pad=SamePad(),
            init_weight=truncated_normal(std=0.01), use_bias=false))

    if model_type === :skipdeq
        init = Conv((3, 3), 64 => 64, gelu; stride=1, pad=SamePad())
    elseif model_type === :regdeq
        init = nothing
    else
        init = missing
    end

    deq = DeepEquilibriumNetwork(deq_model, solver; init, verbose=false,
        linsolve_kwargs=(; maxiters=10), maxiters=10)

    classifier = Chain(
        GroupNorm(64, 64, relu), GlobalMeanPool(), FlattenLayer(), Dense(64, 10))

    model = Chain(; down, deq, classifier)

    # For NVIDIA GPUs this directly generates the parameters on the GPU
    rng = Random.default_rng() |> gdev
    ps, st = Lux.setup(rng, model)

    # Warmup the forward and backward passes
    x = randn(rng, Float32, 28, 28, 1, 2)
    y = onehotbatch(rand(Random.default_rng(), 0:9, 2), 0:9) |> gdev

    @printf "[%s] warming up forward pass\n" string(now())
    loss_function(model, ps, st, (x, y))
    @printf "[%s] warming up backward pass\n" string(now())
    Zygote.gradient(first ∘ loss_function, model, ps, st, (x, y))
    @printf "[%s] warmup complete\n" string(now())

    return model, ps, st
end
```

Define some helper functions to train the model.

```@example basic_mnist_deq
const logit_cross_entropy = CrossEntropyLoss(; logits=Val(true))
const mse_loss = MSELoss()

function loss_function(model, ps, st, (x, y))
    ŷ, st = model(x, ps, st)
    l1 = logit_cross_entropy(ŷ, y)
    l2 = mse_loss(st.deq.solution.z_star, st.deq.solution.u0) # Add in some regularization
    return l1 + eltype(l2)(0.01) * l2, st, (;)
end

function accuracy(model, ps, st, dataloader)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    for (x, y) in dataloader
        target_class = onecold(y)
        predicted_class = onecold(first(model(x, ps, st)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

function train_model(solver, model_type)
    model, ps, st = construct_model(solver; model_type)

    train_dataloader, test_dataloader = loadmnist(32, 0.8) |> gdev

    tstate = Training.TrainState(model, ps, st, Adam(0.0005))

    @printf "[%s] Training Model: %s with Solver: %s\n" string(now()) model_type nameof(typeof(solver))

    @printf "[%s] Pretrain with unrolling to a depth of 5\n" string(now())
    @set! tstate.states = Lux.update_state(tstate.states, :fixed_depth, Val(5))

    for _ in 1:2, (i, (x, y)) in enumerate(train_dataloader)
        _, loss, _, tstate = Training.single_train_step!(
            AutoZygote(), loss_function, (x, y), tstate)
        if i % 10 == 1
            @printf "[%s] Pretraining Batch: [%4d/%4d] Loss: %.5f\n" string(now()) i length(train_dataloader) loss
        end
    end

    acc = accuracy(model, tstate.parameters, tstate.states, test_dataloader) * 100
    @printf "[%s] Pretraining complete. Accuracy: %.5f%%\n" string(now()) acc

    @set! tstate.states = Lux.update_state(tstate.states, :fixed_depth, Val(0))

    for epoch in 1:3
        for (i, (x, y)) in enumerate(train_dataloader)
            _, loss, _, tstate = Training.single_train_step!(
                AutoZygote(), loss_function, (x, y), tstate)
            if i % 10 == 1
                @printf "[%s] Epoch: [%d/%d] Batch: [%4d/%4d] Loss: %.5f\n" string(now()) epoch 3 i length(train_dataloader) loss
            end
        end

        acc = accuracy(model, tstate.parameters, tstate.states, test_dataloader) * 100
        @printf "[%s] Epoch: [%d/%d] Accuracy: %.5f%%\n" string(now()) epoch 3 acc
    end

    @printf "[%s] Training complete.\n" string(now())

    return model, ps, tstate.states
end
```

Now we can train our model. First we will train a Discrete DEQ, which effectively means
pass in a root finding algorithm. Typically most packages lack good nonlinear solvers,
and end up using solvers like `Broyden`, but we can simply slap in any of the fancy solvers
from NonlinearSolve.jl. Here we will use Newton-Krylov Method:

```@example basic_mnist_deq
train_model(NewtonRaphson(; linsolve=KrylovJL_GMRES()), :regdeq);
nothing # hide
```

We can also train a continuous DEQ by passing in an ODE solver. Here we will use `VCAB3()`
which tend to be quite fast for continuous Neural Network problems.

```@example basic_mnist_deq
train_model(VCAB3(), :deq);
nothing # hide
```

This code is setup to allow playing around with different DEQ models. Try modifying the
`model_type` argument to `train_model` to `:skipdeq` or `:deq` to see how the model
behaves. You can also try different solvers from NonlinearSolve.jl and OrdinaryDiffEq.jl!
Even 3rd party solvers from Sundials.jl will work, just remember to use CPU for those.
