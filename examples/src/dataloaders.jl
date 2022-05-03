struct MLDatasetsImageData
    images
    labels
end

MLDatasetsImageData(images::AbstractArray{T,4}, labels::AbstractArray{T,2}) where {T} =
    MLDatasetsImageData(collect(eachslice(images, dims=4)), collect(eachslice(labels, dims=2)))

Base.length(d::MLDatasetsImageData) = length(d.images)
Base.getindex(d::MLDatasetsImageData, i::Int) = (d.images[i], d.labels[i])

function get_dataloaders(
    dataset::Symbol; μ=nothing, σ²=nothing, train_batchsize::Int64, eval_batchsize::Int64
)
    (x_train, y_train), (x_test, y_test), μ, σ², nclasses = if dataset == :CIFAR10
        μ = μ === nothing ? reshape([0.4914f0, 0.4822f0, 0.4465f0], 1, 1, :, 1) : μ
        σ² = σ² === nothing ? reshape([0.2023f0, 0.1994f0, 0.2010f0], 1, 1, :, 1) : σ²
        CIFAR10.traindata(Float32), CIFAR10.testdata(Float32), μ, σ², 10
    else
        throw(ArgumentError("Not yet implemented for $dataset"))
    end

    x_train = (x_train .- μ) ./ σ²
    y_train = Float32.(onehotbatch(y_train, 0:(nclasses - 1)))
    x_test = (x_test .- μ) ./ σ²
    y_test = Float32.(onehotbatch(y_test, 0:(nclasses - 1)))

    train_dataset = shuffleobs(MLDatasetsImageData(x_train, y_train))
    train_dataset = is_distributed() ? DistributedDataContainer(train_dataset) : train_dataset
    test_dataset = MLDatasetsImageData(x_test, y_test)
    test_dataset = is_distributed() ? DistributedDataContainer(test_dataset) : test_dataset

    return (DataLoader(train_dataset, train_batchsize), DataLoader(test_dataset, eval_batchsize))
end
