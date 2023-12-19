# Training a Simple MNIST Classifier using Deep Equilibrium Models

We will train a simple Deep Equilibrium Model on MNIST. First we load a few packages.

```@example basic_mnist_deq
using DeepEquilibriumNetworks, SciMLSensitivity, Lux, NonlinearSolve, OrdinaryDiffEq,
    Statistics, Random, Optimisers, LuxCUDA, Zygote, LinearSolve
using MLDatasets: MNIST
using MLDataUtils: LabelEnc, convertlabel, stratifiedobs, batchview

CUDA.allowscalar(false)
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
```

Setup device functions from Lux. See
[GPU Management](https://lux.csail.mit.edu/dev/manual/gpu_management) for more details.

```@example basic_mnist_deq
const cdev = cpu_device()
const gdev = gpu_device()
```

We can now construct our dataloader.

```@example basic_mnist_deq
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

Construct the Lux Neural Network containing a DEQ layer.

```@example basic_mnist_deq
function construct_model(solver; model_type::Symbol=:deq)
    down = Chain(Conv((3, 3), 1 => 64, gelu; stride=1), GroupNorm(64, 64),
        Conv((4, 4), 64 => 64; stride=2, pad=1))

    # The input layer of the DEQ
    deq_model = Chain(Parallel(+,
            Conv((3, 3), 64 => 64, tanh; stride=1, pad=SamePad()),
            Conv((3, 3), 64 => 64, tanh; stride=1, pad=SamePad())),
        Conv((3, 3), 64 => 64, tanh; stride=1, pad=SamePad()))

    if model_type === :skipdeq
        init = Conv((3, 3), 64 => 64, gelu; stride=1, pad=SamePad())
    elseif model_type === :regdeq
        init = nothing
    else
        init = missing
    end

    deq = DeepEquilibriumNetwork(deq_model, solver; init, verbose=false,
        linsolve_kwargs=(; maxiters=10))

    classifier = Chain(GroupNorm(64, 64, relu), GlobalMeanPool(), FlattenLayer(),
        Dense(64, 10))

    model = Chain(; down, deq, classifier)

    # For NVIDIA GPUs this directly generates the parameters on the GPU
    rng = Random.default_rng() |> gdev
    ps, st = Lux.setup(rng, model)

    # Warmup the forward and backward passes
    x = randn(rng, Float32, 28, 28, 1, 128)
    y = onehot(rand(Random.default_rng(), 0:9, 128)) |> gdev

    model_ = Lux.Experimental.StatefulLuxLayer(model, ps, st)
    @info "warming up forward pass"
    logitcrossentropy(model_, x, ps, y)
    @info "warming up backward pass"
    Zygote.gradient(logitcrossentropy, model_, x, ps, y)
    @info "warmup complete"

    return model, ps, st
end
```

Define some helper functions to train the model.

```@example basic_mnist_deq
logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims=1); dims=1))
function logitcrossentropy(model, x, ps, y)
    l1 = logitcrossentropy(model(x, ps), y)
    # Add in some regularization
    l2 = mean(abs2, model.st.deq.solution.z_star .- model.st.deq.solution.u0)
    return l1 + 10.0 * l2
end

classify(x) = argmax.(eachcol(x))

function accuracy(model, data, ps, st)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    model = Lux.Experimental.StatefulLuxLayer(model, ps, st)
    for (x, y) in data
        target_class = classify(cdev(y))
        predicted_class = classify(cdev(model(x)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

function train_model(solver, model_type; data_train=zip(x_train, y_train),
        data_test=zip(x_test, y_test))
    model, ps, st = construct_model(solver; model_type)
    model_st = Lux.Experimental.StatefulLuxLayer(model, nothing, st)

    @info "Training Model: $(model_type) with Solver: $(nameof(typeof(solver)))"

    opt_st = Optimisers.setup(Adam(0.001), ps)

    acc = accuracy(model, data_test, ps, st) * 100
    @info "Starting Accuracy: $(acc)"

    # = Uncomment these lines to enavle pretraining. See what happens
    @info "Pretrain with unrolling to a depth of 5"
    st = Lux.update_state(st, :fixed_depth, Val(5))
    model_st = Lux.Experimental.StatefulLuxLayer(model, ps, st)

    for (i, (x, y)) in enumerate(data_train)
        res = Zygote.withgradient(logitcrossentropy, model_st, x, ps, y)
        Optimisers.update!(opt_st, ps, res.grad[3])
        if i % 50 == 1
            @info "Pretraining Batch: [$(i)/$(length(data_train))] Loss: $(res.val)"
        end
    end

    acc = accuracy(model, data_test, ps, model_st.st) * 100
    @info "Pretraining complete. Accuracy: $(acc)"
    # =#

    st = Lux.update_state(st, :fixed_depth, Val(0))
    model_st = Lux.Experimental.StatefulLuxLayer(model, ps, st)

    for epoch in 1:3
        for (i, (x, y)) in enumerate(data_train)
            res = Zygote.withgradient(logitcrossentropy, model_st, x, ps, y)
            Optimisers.update!(opt_st, ps, res.grad[3])
            if i % 50 == 1
                @info "Epoch: [$(epoch)/3] Batch: [$(i)/$(length(data_train))] Loss: $(res.val)"
            end
        end

        acc = accuracy(model, data_test, ps, model_st.st) * 100
        @info "Epoch: [$(epoch)/3] Accuracy: $(acc)"
    end

    @info "Training complete."
    println()

    return model, ps, st
end
```

Now we can train our model. First we will train a Discrete DEQ, which effectively means
pass in a root finding algorithm. Typically most packages lack good nonlinear solvers,
and end up using solvers like `Broyden`, but we can simply slap in any of the fancy solvers
from NonlinearSolve.jl. Here we will use Newton-Krylov Method:

```@example basic_mnist_deq
train_model(NewtonRaphson(; linsolve=KrylovJL_GMRES()), :regdeq)
nothing # hide
```

We can also train a continuous DEQ by passing in an ODE solver. Here we will use `VCAB3()`
which tend to be quite fast for continuous Neural Network problems.

```@example basic_mnist_deq
train_model(VCAB3(), :deq)
nothing # hide
```

This code is setup to allow playing around with different DEQ models. Try modifying the
`model_type` argument to `train_model` to `:skipdeq` or `:deq` to see how the model
behaves. You can also try different solvers from NonlinearSolve.jl and OrdinaryDiffEq.jl!
Even 3rd party solvers from Sundials.jl will work, just remember to use CPU for those.
