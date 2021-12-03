using ChemistryFeaturization, CSV, CUDA, DataFrames, Dates, FastDEQ, Flux, Random, Serialization, Statistics,
      SteadyStateDiffEq, Wandb, Zygote
using ParameterSchedulers: Scheduler, Cos

const DATA_PATH = joinpath(@__DIR__, "data", "qm9")
const CACHED_DATASET = Dict()

function load_dataset(data_dir=DATA_PATH; num_data_points::Union{Int,Nothing}=nothing,
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

    # pick out train/test sets
    verbose && @info("Dividing into train/test sets...")
    train_output = output[1:num_train]
    test_output = output[(num_train + 1):end]
    train_input = inputs[1:num_train]
    test_input = inputs[(num_train + 1):end]

    return (train_input, train_output), (test_input, test_output)
end

function construct_dataiterators(X, y; batch_size::Int=128)
    return zip(BatchedAtomicGraph(batch_size, X), Iterators.partition(y, batch_size))
end

function get_model(model_type::Symbol; num_features::Int=61, num_conv::Int=15, crystal_feature_length::Int=128,
                   num_hidden_layers::Int=5, abstol::Real=0.1f0, reltol::Real=0.1f0, maxiters::Int=10)
    model = CrystalGraphCNN(num_features; num_conv=num_conv, atom_conv_feature_length=crystal_feature_length,
                            pooled_feature_length=crystal_feature_length ÷ 2, num_hidden_layers=num_hidden_layers,
                            deq_type=model_type, abstol=Float32(abstol), reltol=Float32(reltol), maxiters=maxiters)

    return gpu(model)
end

function register_nfe_counts(model, buffer)
    callback() = push!(buffer, get_and_clear_nfe!(model))
    return callback
end

function compute_total_loss(model, dataloader)
    total_loss, total_datasize, total_nfe = 0, 0, 0
    for (x, y) in dataloader
        x = gpu(x)
        y = gpu(y)

        ŷ = model(x)
        ŷ = ŷ isa Tuple ? ŷ[1] : ŷ
        total_nfe += get_and_clear_nfe!(model) * length(y)
        total_loss += mean(abs2, vec(ŷ) .- y) * length(y)
        total_datasize += length(y)
    end
    return (total_loss / total_datasize, total_nfe / total_datasize)
end

function train(config::Dict)
    ## Setup Logging & Experiment Configuration
    expt_name = "fastdeqjl-qm9_formation_energy-deq-$(now())"
    lg_wandb = WandbLogger(; project="FastDEQ.jl", name=expt_name, config=config)
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

    train_dataloader = construct_dataiterators(train_input, train_output; batch_size=batch_size)
    val_dataloader = construct_dataiterators(val_input, val_output; batch_size=eval_batch_size)

    ## Model Setup
    model = get_model(Symbol(get_config(lg_wandb, "model_type")); abstol=get_config(lg_wandb, "abstol"),
                      reltol=get_config(lg_wandb, "reltol"), maxiters=get_config(lg_wandb, "maxiters"))

    loss_function = SupervisedLossContainer((ŷ, y) -> mean(abs2, y .- vec(ŷ)), 2.5f0)

    ## Warmup
    __x, __y = first(train_dataloader) .|> gpu
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
            train_start_time = time()
            for (x, y) in train_dataloader
                x = gpu(x)
                y = gpu(y)

                _res = Zygote.withgradient(() -> loss_function(model, x, y), ps)
                loss = _res.val
                gs = _res.grad
                Flux.Optimise.update!(opt, ps, gs)

                ### Store the NFE Count
                cb()

                ### Log the losses
                log(lg_wandb,
                    Dict("Training/Step/Loss" => loss, "Training/Step/NFE" => nfe_counts[end],
                         "Training/Step/Count" => step))
                lg_term(; records=Dict("Train/Running/NFE" => nfe_counts[end], "Train/Running/Loss" => loss))
                step += 1
            end
            train_end_time = time()

            ### Validation Loss
            val_start_time = time()
            val_loss, val_nfe = compute_total_loss(model, val_dataloader)
            val_end_time = time()

            log(lg_wandb,
                Dict("Validation/Epoch/Count" => epoch, "Validation/Epoch/Loss" => val_loss, "Validation/Epoch/NFE" => val_nfe))
            lg_term(epoch, train_end_time - train_start_time, val_nfe, val_loss, val_end_time - val_start_time)
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

        config = Dict("seed" => seed, "learning_rate" => 0.005, "abstol" => 0.1f0, "reltol" => 0.1f0, "maxiters" => 15,
                      "epochs" => 10, "batch_size" => 512, "eval_batch_size" => 512, "model_type" => model_type,
                      "solver_type" => "dynamicss", "num_data_points" => 133885)

        model, nfe_counts = train(config)

        push!(nfe_count_dict[model_type], nfe_counts)
    end
end
