# Comparing SkipDEQ and DEQ on a Polynomial Fitting Problem

## Load Packages
using Dates, FastDEQ, Statistics, Plots, Random, Wandb

CUDA.allowscalar(false)
enable_fast_mode!()

## Model and Loss Function
function get_model(hdims::Int, abstol::T, reltol::T, model_type::String) where {T}
    main_model = Chain(Parallel((x₁, x₂) -> gelu.(x₁ .+ x₂), Dense(hdims, hdims * 2), Dense(hdims, hdims * 2)),
                       Dense(hdims * 2, hdims))
    aux_model = Chain(Dense(hdims, hdims * 5, gelu), Dense(hdims * 5, hdims))
    args = model_type == "vanilla" ? (main_model,) : (model_type == "skip" ? (main_model, aux_model) : (main_model,))
    _deq = model_type == "vanilla" ? DeepEquilibriumNetwork : SkipDeepEquilibriumNetwork

    return gpu(DEQChain(Dense(1, hdims, gelu),
                        _deq(args..., get_default_dynamicss_solver(Float32(abstol), Float32(reltol));
                             sensealg=get_default_ssadjoint(Float32(abstol), Float32(reltol), 50), maxiters=50,
                             verbose=false), Dense(hdims, 1)))
end

## Utilities
register_nfe_counts(deq, buffer) = () -> push!(buffer, get_and_clear_nfe!(deq))

## Training Function
function train(config::Dict)
    ## Setup Logging & Experiment Configuration
    lg = WandbLogger(; project="FastDEQ.jl", name="fastdeqjl-polynomial_fitting-$(now())", config=config)

    ## Reproducibility
    Random.seed!(get_config(lg, "seed"))

    ## Data Generation
    batch_size = get_config(lg, "batch_size")
    x_data = gpu(rand(Float32, 1, get_config(lg, "data_size")) .* 2 .- 1)
    y_data = gpu(x_data .^ 2)

    batch_idxs = Iterators.partition(1:size(x_data, 2), batch_size)
    x_data_partition = (x_data[:, i] for i in batch_idxs)
    y_data_partition = (y_data[:, i] for i in batch_idxs)

    ## Model Setup
    model = get_model(get_config(lg, "hidden_dims"), get_config(lg, "abstol"), get_config(lg, "reltol"),
                      get_config(lg, "model_type"))

    loss_function = SupervisedLossContainer(Flux.Losses.mse, 1.0f-2)

    nfe_counts = Int64[]
    cb = register_nfe_counts(model, nfe_counts)

    ## Training Loop
    ps = Flux.params(model)
    opt = ADAMW(get_config(lg, "learning_rate"), (0.9, 0.999), 0.001)
    step = 1
    for epoch in 1:get_config(lg, "epochs")
        try
            epoch_loss = 0.0f0
            epoch_nfe = 0
            for (x, y) in zip(x_data_partition, y_data_partition)
                loss, back = Zygote.pullback(() -> loss_function(model, x, y), ps)
                gs = back(one(loss))
                Flux.Optimise.update!(opt, ps, gs)

                ### Clear the NFE Count
                get_and_clear_nfe!(model)

                ### Log the losses
                ŷ = model(x)
                loss = loss_function.loss_function(ŷ isa Tuple ? ŷ[1] : ŷ, y)
                cb()

                log(lg,
                    Dict("Training/Step/Loss" => loss, "Training/Step/NFE" => nfe_counts[end],
                         "Training/Step/Count" => step))
                step += 1
                epoch_nfe += nfe_counts[end] * size(x, ndims(x))
                epoch_loss += loss * size(x, ndims(x))
            end
            ### Log the epoch loss
            epoch_loss /= size(x_data, ndims(x_data))
            epoch_nfe /= size(x_data, ndims(x_data))
            log(lg,
                Dict("Training/Epoch/Loss" => epoch_loss, "Training/Epoch/NFE" => epoch_nfe,
                     "Training/Epoch/Count" => epoch))
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

    close(lg)

    return model, nfe_counts, x_data, y_data
end

## Run the experiment
for seed in [1, 11, 111]
    for model_type in ["vanilla", "skip", "skip_no_extra_param"]
        config = Dict("seed" => seed, "learning_rate" => 1.0f-4, "abstol" => 1.0f-3, "reltol" => 1.0f-3,
                      "epochs" => 250, "batch_size" => 128, "data_size" => 512, "hidden_dims" => 50,
                      "model_type" => model_type)

        model, nfe_counts, x_data, y_data = train(config)
    end
end
