using CSV, DataFrames
using Random, Statistics
using Flux
using Flux: @epochs
using ChemistryFeaturization
using Serialization
using FastDEQ

function train(;
    num_pts = 133885,  # Total Pts = 133885
    num_epochs = 100,
    data_dir = joinpath(@__DIR__, "data"),
    verbose = true,
    batch_size = 128,
)
    verbose && println("Setting things up...")

    # data-related options
    train_frac = 0.8 # what fraction for training?
    num_train = Int32(round(train_frac * num_pts))
    num_test = num_pts - num_train
    prop = :Cv # choose any column from labels.csv except :key
    id = :key # field by which to label each input material

    # model hyperparameters – keeping it pretty simple for now
    num_conv = 5 # how many convolutional layers?
    crys_fea_len = 128 # length of crystal feature vector after pooling (keep node dimension constant for now)
    num_hidden_layers = 5 # how many fully-connected layers after convolution and pooling?
    opt = ADAM(0.001) # optimizer

    # dataset...first, read in outputs
    info = CSV.read(joinpath(data_dir, "labels_qm9.csv"), DataFrame)
    y = reshape(Array(Float32.(info[!, prop])), 1, :)

    # shuffle data and pick out subset
    indices = shuffle(1:size(info, 1))[1:num_pts]
    info = info[indices, :]
    output = y[indices]

    # next, read in prefeaturized graphs
    verbose && println("Reading in graphs...")

    inputs = FeaturizedAtoms[]

    for r in eachrow(info)
        fpath = joinpath(data_dir, "qm9_jls", "$(r[id]).jls")
        push!(inputs, deserialize(fpath))
    end

    # pick out train/test sets
    verbose && println("Dividing into train/test sets...")
    train_output = output[1:num_train] |> gpu
    test_output = output[num_train+1:end] |> gpu
    train_input = inputs[1:num_train]
    test_input = inputs[num_train+1:end]

    # 2 gpu calls to remove sparsity (some bugs with sparse dense matmul)
    train_input = BatchedAtomicGraph(batch_size, train_input) .|> gpu .|> gpu
    test_input = BatchedAtomicGraph(batch_size, test_input) .|> gpu .|> gpu

    train_data = zip(train_input, Iterators.partition(train_output, batch_size))
    test_data = zip(test_input, Iterators.partition(test_output, batch_size))

    # build the model
    verbose && println("Building the network...")
    num_features = size(inputs[1].encoded_features, 1)
    model = CrystalGraphCNN(
        num_features,
        num_conv = num_conv,
        atom_conv_feature_length = crys_fea_len,
        pooled_feature_length = (Int(crys_fea_len / 2)),
        num_hidden_layers = num_hidden_layers,
    ) |> gpu

    # define loss function and a callback to monitor progress
    loss(x, y) = Flux.Losses.mse(vec(model(x)), y)
    loss(xy) = loss(xy[1], xy[2])
    evalcb_verbose() = @show(mean(loss.(test_data)))
    evalcb_quiet() = return nothing
    evalcb = verbose ? evalcb_verbose : evalcb_quiet
    evalcb()

    # train
    verbose && println("Training!")
    @epochs num_epochs Flux.train!(
        loss,
        Flux.params(model),
        train_data,
        opt,
        cb = Flux.throttle(evalcb, 5),
    )

    return model
end
