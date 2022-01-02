using ChemistryFeaturization, CSV, CUDA, DataFrames, Dates, FastDEQ, FluxExperimental, FluxMPI, FileIO, JLD2, Random,
      Serialization, Statistics, Wandb, Zygote
using ParameterSchedulers: Scheduler, Cos

CUDA.versioninfo()

FluxMPI.Init()
const DATA_PATH = joinpath(@__DIR__, "data", "qm9")
const CACHED_DATASET = Dict()

const MPI_COMM_WORLD = MPI.COMM_WORLD
const MPI_COMM_SIZE = MPI.Comm_size(MPI_COMM_WORLD)

function load_dataset_from_serialized_files(data_dir=DATA_PATH; num_data_points::Union{Int,Nothing}=nothing,
                                            train_fraction::Float64=0.8, verbose::Bool=false, seed::Int=0)
    Random.seed!(seed)

    num_data_points = isnothing(num_data_points) ? 133885 : num_data_points
    num_train = Int32(round(train_fraction * num_data_points))
    prop = :Cv # choose any column from labels.csv except :key
    id = :key # field by which to label each input material

    info = CSV.read(joinpath(data_dir, "labels_qm9.csv"), DataFrame)
    y = reshape(Array(Float32.(info[!, prop])), 1, :)

    # shuffle data and pick out subset
    indices = shuffle(1:size(info, 1))[1:num_data_points]
    info = info[indices, :]
    output = y[indices]

    # next, read in prefeaturized graphs
    verbose && @info("Reading in graphs...")

    inputs = Vector{FeaturizedAtoms}(undef, num_data_points)

    Threads.@threads for i in 1:length(indices)
        r = info[i, id]
        fpath = joinpath(data_dir, "qm9_jls", "$(r).jls")
        inputs[i] = deserialize(fpath)
    end

    # pick out train/val sets
    verbose && @info("Dividing into train/val sets...")
    train_output = output[1:num_train]
    val_output = output[(num_train + 1):end]
    train_input = inputs[1:num_train]
    val_input = inputs[(num_train + 1):end]

    return (train_input, train_output), (val_input, val_output)
end

function load_dataset(data_dir=DATA_PATH; kwargs...)
    filepath = joinpath(data_dir, "qm9_data.jld2")
    if !isfile(filepath)
        @info "JLD2 file not found. Attempting to load from serialized files..."
        return load_dataset_from_serialized_files(data_dir; kwargs...)
    end

    @info "Loading JLD2 file..."
    @info "All kwargs are ignored since JLD2 file found..."

    data = load(filepath)
    return (data["train_X"], data["train_y"]), (data["val_X"], data["val_y"])
end

function construct_dataiterators(X, y; batch_size::Int=128, shuffle::Bool=false)
    return DataParallelDataLoader((BatchedAtomicGraph(batch_size, X), collect(Iterators.partition(y, batch_size)));
                                  batchsize=1, shuffle=shuffle)
end

function get_model(model_type::Symbol; num_features::Int=61, crystal_feature_length::Int=128, num_hidden_layers::Int=5,
                   abstol::Real=0.1f0, reltol::Real=0.1f0, maxiters::Int=10)
    model = CrystalGraphCNN(num_features; num_conv=maxiters + 1, atom_conv_feature_length=crystal_feature_length,
                            pooled_feature_length=crystal_feature_length ÷ 2, num_hidden_layers=num_hidden_layers,
                            deq_type=model_type, abstol=Float32(abstol), reltol=Float32(reltol), maxiters=maxiters)

    return (MPI_COMM_SIZE > 1 ? DataParallelFluxModel : gpu)(model)
end

register_nfe_counts(deq, buffer) = () -> push!(buffer, get_and_clear_nfe!(deq))

function compute_total_loss(model, dataloader, rescale)
    total_loss, total_datasize, total_nfe = 0, 0, 0
    for (x, y) in dataloader
        x = gpu(x[1])
        y = gpu(y[1])

        ŷ = model(x)
        ŷ = ŷ isa Tuple ? ŷ[1] : ŷ
        ŷ = rescale(σ.(vec(ŷ)))
        total_nfe += get_and_clear_nfe!(model) * length(y)
        total_loss += mean(abs, ŷ .- y) * length(y)
        total_datasize += length(y)
    end
    return (total_loss / total_datasize, total_nfe / total_datasize)
end

function train(config::Dict)
    ## Setup Logging & Experiment Configuration
    expt_name = "fastdeqjl-qm9_formation_energy-deq-$(now())"
    lg_wandb = WandbLoggerMPI(; project="FastDEQ.jl", name=expt_name, config=config)
    lg_term = PrettyTableLogger("logs/" * expt_name * ".log",
                                ["Epoch Number", "Train/Time", "Validation/NFE", "Validation/Loss", "Validation/Time"],
                                ["Train/Running/NFE", "Train/Running/Loss"])

    ## Reproducibility
    Random.seed!(get_config(lg_wandb, "seed"))

    ## Dataset
    batch_size = get_config(lg_wandb, "batch_size")
    eval_batch_size = get_config(lg_wandb, "eval_batch_size")
    num_data_points = get_config(lg_wandb, "num_data_points")

    k = string(get_config(lg_wandb, "seed")) * "_" * string(num_data_points)
    (train_input, train_output), (val_input, val_output) = if length(CACHED_DATASET) == 0
        res = load_dataset(; num_data_points=num_data_points, verbose=true, seed=get_config(lg_wandb, "seed"))
        CACHED_DATASET[k] = res
        @info "Cached the Loaded Dataset"
        res
    else
        if k in keys(CACHED_DATASET)
            CACHED_DATASET[k]
        else
            res = load_dataset(; num_data_points=num_data_points, verbose=true, seed=get_config(lg_wandb, "seed"))
            CACHED_DATASET[k] = res
            @info "Cached the Loaded Dataset"
            res
        end
    end

    min_y, max_y = CUDA.@allowscalar extrema(train_output)
    rescale(v) = v .* (max_y - min_y) .+ min_y

    train_dataloader = construct_dataiterators(train_input, train_output; batch_size=batch_size)
    val_dataloader = construct_dataiterators(val_input, val_output; batch_size=eval_batch_size)

    ## Model Setup
    model = get_model(Symbol(get_config(lg_wandb, "model_type")); abstol=get_config(lg_wandb, "abstol"),
                      reltol=get_config(lg_wandb, "reltol"), maxiters=get_config(lg_wandb, "maxiters"))

    loss_function = SupervisedLossContainer((ŷ, y) -> mean(abs, y .- rescale(σ.(vec(ŷ)))), 2.5f0, 0.0f0,
                                            0.0f0)

    ## Warmup
    __x, __y = gpu.(first.(first(train_dataloader)))
    loss_function(model, __x, __y)
    @info "Forward Pass Warmup Completed"
    Zygote.gradient(() -> loss_function(model, __x, __y), Flux.params(model))
    @info "Backward Pass Warmup Completed"

    nfe_counts = []
    cb = register_nfe_counts(model, nfe_counts)

    ## Training Loop
    ps = Flux.params(model)
    opt = Scheduler(Cos(get_config(lg_wandb, "learning_rate"), 1e-6,
                        length(train_dataloader) * get_config(lg_wandb, "epochs")),
                    ADAM(get_config(lg_wandb, "learning_rate"), (0.9, 0.999)))
    step = 1

    for epoch in 1:get_config(lg_wandb, "epochs")
        try
            train_time = 0
            for (x, y) in train_dataloader
                x = gpu(x[1])
                y = gpu(y[1])

                train_start_time = time()
                _res = Zygote.withgradient(() -> loss_function(model, x, y), ps)
                loss = _res.val
                gs = _res.grad
                Flux.Optimise.update!(opt, ps, gs)
                train_time += time() - train_start_time

                ### Store the NFE Count
                cb()

                ### Log the losses
                log(lg_wandb,
                    Dict("Training/Step/Loss" => loss, "Training/Step/NFE" => nfe_counts[end],
                         "Training/Step/Count" => step))
                lg_term(; records=Dict("Train/Running/NFE" => nfe_counts[end], "Train/Running/Loss" => loss))
                step += 1
            end

            ### Validation Loss
            val_start_time = time()
            val_loss, val_nfe = compute_total_loss(model, val_dataloader, rescale)
            val_end_time = time()

            log(lg_wandb,
                Dict("Validation/Epoch/Count" => epoch, "Validation/Epoch/Loss" => val_loss,
                     "Validation/Epoch/NFE" => val_nfe))
            lg_term(epoch, train_time, val_nfe, val_loss, val_end_time - val_start_time)
        catch ex
            if ex isa Flux.Optimise.StopException
                break
            elseif ex isa Flux.Optimise.SkipException
                continue
            else
                rethrow(ex)
            end
        end
    end

    close(lg_wandb)
    close(lg_term)

    return model, nfe_counts
end

## Run Experiment
nfe_count_dict = Dict("explicit" => [], "deq" => [], "skip_deq" => [], "skip_deq_no_extra_params" => [])

for seed in [1, 11, 111]
    for model_type in ["explicit", "deq", "skip_deq", "skip_deq_no_extra_params"]
        @info "Starting Run for Model: $model_type with Seed: $seed"

        config = Dict("seed" => seed, "learning_rate" => 0.0001, "abstol" => 0.1f0, "reltol" => 0.1f0, "maxiters" => 50,
                      "epochs" => 100, "batch_size" => 32, "eval_batch_size" => 32, "model_type" => model_type,
                      "solver_type" => "dynamicss", "num_data_points" => 133885)

        model, nfe_counts = train(config)

        push!(nfe_count_dict[model_type], nfe_counts)
    end
end
