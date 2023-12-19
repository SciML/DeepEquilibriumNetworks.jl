# Training a Simple MNIST Classifier using Deep Equilibrium Models

We will train a simple Deep Equilibrium Model on MNIST. First we load a few packages.

```@example basic_mnist_deq
using DeepEquilibriumNetworks, SciMLSensitivity, Lux, NonlinearSolve, OrdinaryDiffEq,
    Statistics, Random, Optimization, OptimizationOptimisers, LuxCUDA
using MLDatasets: MNIST
using MLDataUtils: LabelEnc, convertlabel, stratifiedobs

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

function loadmnist(batchsize)
    # Load MNIST
    mnist = MNIST(; split=:train)
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
```
