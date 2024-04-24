# Modelling Equilibrium Models with Reduced State Size

Sometimes we want don't want to solve a root finding problem with the full state size. This
will often be faster, since the size of the root finding problem is reduced. We will use the
same MNIST example as before, but this time we will use a reduced state size.

```@example reduced_dim_mnist
using DeepEquilibriumNetworks, SciMLSensitivity, Lux, NonlinearSolve, OrdinaryDiffEq,
      Statistics, Random, Optimisers, LuxCUDA, Zygote, LinearSolve, Dates, Printf
using MLDatasets: MNIST
using MLDataUtils: LabelEnc, convertlabel, stratifiedobs, batchview

CUDA.allowscalar(false)
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

const cdev = cpu_device()
const gdev = gpu_device()

function onehot(labels_raw)
    return convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
end

function loadmnist(batchsize, split)
    # Load MNIST
    mnist = MNIST(; split)
    imgs, labels_raw = mnist.features, mnist.targets
    # Process images into (H,W,C,BS) batches
    x_train = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3))) |>
              gdev
    x_train = batchview(x_train, batchsize)
    # Onehot and batch the labels
    y_train = onehot(labels_raw) |> gdev
    y_train = batchview(y_train, batchsize)
    return x_train, y_train
end

x_train, y_train = loadmnist(128, :train);
x_test, y_test = loadmnist(128, :test);
```

Now we will define the construct model function. Here we will use Dense Layers and
downsample the features using the `init` kwarg.

```@example reduced_dim_mnist
function construct_model(solver; model_type::Symbol=:regdeq)
    down = Chain(FlattenLayer(), Dense(784 => 512, gelu))

    # The input layer of the DEQ
    deq_model = Chain(Parallel(+, Dense(128 => 64, tanh),   # Reduced dim of `128`
            Dense(512 => 64, tanh)),  # Original dim of `512`
        Dense(64 => 64, tanh), Dense(64 => 128))       # Return the reduced dim of `128`

    if model_type === :skipdeq
        init = Dense(512 => 128, tanh)
    elseif model_type === :regdeq
        error(":regdeq is not supported for reduced dim models")
    else
        # This should preferably done via `ChainRulesCore.@ignore_derivatives`. But here
        # we are only using Zygote so this is fine.
        init = WrappedFunction(x -> Zygote.@ignore(fill!(
            similar(x, 128, size(x, 2)), false)))
    end

    deq = DeepEquilibriumNetwork(
        deq_model, solver; init, verbose=false, linsolve_kwargs=(; maxiters=10))

    classifier = Chain(Dense(128 => 128, gelu), Dense(128, 10))

    model = Chain(; down, deq, classifier)

    # For NVIDIA GPUs this directly generates the parameters on the GPU
    rng = Random.default_rng() |> gdev
    ps, st = Lux.setup(rng, model)

    # Warmup the forward and backward passes
    x = randn(rng, Float32, 28, 28, 1, 128)
    y = onehot(rand(Random.default_rng(), 0:9, 128)) |> gdev

    model_ = StatefulLuxLayer(model, ps, st)
    @printf "[%s] warming up forward pass\n" string(now())
    logitcrossentropy(model_, x, ps, y)
    @printf "[%s] warming up backward pass\n" string(now())
    Zygote.gradient(logitcrossentropy, model_, x, ps, y)
    @printf "[%s] warmup complete\n" string(now())

    return model, ps, st
end
```

Define some helper functions to train the model.

```@example reduced_dim_mnist
logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims=1); dims=1))
function logitcrossentropy(model, x, ps, y)
    l1 = logitcrossentropy(model(x, ps), y)
    # Add in some regularization
    l2 = mean(abs2, model.st.deq.solution.z_star .- model.st.deq.solution.u0)
    return l1 + 0.1f0 * l2
end

classify(x) = argmax.(eachcol(x))

function accuracy(model, data, ps, st)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    model = StatefulLuxLayer(model, ps, st)
    for (x, y) in data
        target_class = classify(cdev(y))
        predicted_class = classify(cdev(model(x)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

function train_model(
        solver, model_type; data_train=zip(x_train, y_train), data_test=zip(x_test, y_test))
    model, ps, st = construct_model(solver; model_type)
    model_st = StatefulLuxLayer(model, nothing, st)

    @printf "[%s] Training Model: %s with Solver: %s\n" string(now()) model_type nameof(typeof(solver))

    opt_st = Optimisers.setup(Adam(0.001), ps)

    acc = accuracy(model, data_test, ps, st) * 100
    @printf "[%s] Starting Accuracy: %.5f%%\n" string(now()) acc

    @printf "[%s] Pretrain with unrolling to a depth of 5\n" string(now())
    st = Lux.update_state(st, :fixed_depth, Val(5))
    model_st = StatefulLuxLayer(model, ps, st)

    for (i, (x, y)) in enumerate(data_train)
        res = Zygote.withgradient(logitcrossentropy, model_st, x, ps, y)
        Optimisers.update!(opt_st, ps, res.grad[3])
        i % 50 == 1 && @printf "[%s] Pretraining Batch: [%4d/%4d] Loss: %.5f\n" string(now()) i length(data_train) res.val
    end

    acc = accuracy(model, data_test, ps, model_st.st) * 100
    @printf "[%s] Pretraining complete. Accuracy: %.5f%%\n" string(now()) acc

    st = Lux.update_state(st, :fixed_depth, Val(0))
    model_st = StatefulLuxLayer(model, ps, st)

    for epoch in 1:3
        for (i, (x, y)) in enumerate(data_train)
            res = Zygote.withgradient(logitcrossentropy, model_st, x, ps, y)
            Optimisers.update!(opt_st, ps, res.grad[3])
            i % 50 == 1 && @printf "[%s] Epoch: [%d/%d] Batch: [%4d/%4d] Loss: %.5f\n" string(now()) epoch 3 i length(data_train) res.val
        end

        acc = accuracy(model, data_test, ps, model_st.st) * 100
        @printf "[%s] Epoch: [%d/%d] Accuracy: %.5f%%\n" string(now()) epoch 3 acc
    end

    @printf "[%s] Training complete.\n" string(now())

    return model, ps, st
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
