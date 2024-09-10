# Modelling Equilibrium Models with Reduced State Size

Sometimes we want don't want to solve a root finding problem with the full state size. This
will often be faster, since the size of the root finding problem is reduced. We will use the
same MNIST example as before, but this time we will use a reduced state size.

```@example reduced_dim_mnist
using DeepEquilibriumNetworks, SciMLSensitivity, Lux, NonlinearSolve, OrdinaryDiffEq,
      Random, Optimisers, Zygote, LinearSolve, Dates, Printf, Setfield, OneHotArrays
using MLDatasets: MNIST
using MLUtils: DataLoader, splitobs
using LuxCUDA # For NVIDIA GPU support

CUDA.allowscalar(false)
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

const cdev = cpu_device()
const gdev = gpu_device()

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

Now we will define the construct model function. Here we will use Dense Layers and
downsample the features using the `init` kwarg.

```@example reduced_dim_mnist
function construct_model(solver; model_type::Symbol=:regdeq)
    down = Chain(FlattenLayer(), Dense(784 => 512, gelu))

    # The input layer of the DEQ
    deq_model = Chain(
        Parallel(+,
            Dense(
                128 => 64, tanh; use_bias=false, init_weight=truncated_normal(; std=0.01)),   # Reduced dim of `128`
            Dense(
                512 => 64, tanh; use_bias=false, init_weight=truncated_normal(; std=0.01))),  # Original dim of `512`
        Dense(64 => 64, tanh; use_bias=false, init_weight=truncated_normal(; std=0.01)),
        Dense(64 => 128; use_bias=false, init_weight=truncated_normal(; std=0.01)))       # Return the reduced dim of `128`

    if model_type === :skipdeq
        init = Dense(
            512 => 128, tanh; use_bias=false, init_weight=truncated_normal(; std=0.01))
    elseif model_type === :regdeq
        error(":regdeq is not supported for reduced dim models")
    else
        # This should preferably done via `ChainRulesCore.@ignore_derivatives`. But here
        # we are only using Zygote so this is fine.
        init = WrappedFunction(x -> Zygote.@ignore(fill!(
            similar(x, 128, size(x, 2)), false)))
    end

    deq = DeepEquilibriumNetwork(deq_model, solver; init, verbose=false,
        linsolve_kwargs=(; maxiters=10), maxiters=10)

    classifier = Chain(Dense(128 => 128, gelu), Dense(128, 10))

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

```@example reduced_dim_mnist
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

Now we can train our model. We can't use `:regdeq` here currently, but we will support this
in the future.

```@example reduced_dim_mnist
train_model(NewtonRaphson(; linsolve=KrylovJL_GMRES()), :skipdeq)
nothing # hide
```

```@example reduced_dim_mnist
train_model(NewtonRaphson(; linsolve=KrylovJL_GMRES()), :deq)
nothing # hide
```
