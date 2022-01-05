# Comparing SkipDEQ and DEQ on a Polynomial Fitting Problem

## Load Packages
using Dates, FastDEQ, Statistics, Plots, Random, Wandb

CUDA.allowscalar(false)
enable_fast_mode!()

## Model and Loss Function
function get_model(hdims::Int, abstol::T, reltol::T, model_type::String, jacobian_regularization::Bool,
                   residual_regularization::Bool) where {T}
    main_model = Chain(Parallel((x₁, x₂) -> relu.(x₁ .+ x₂), Dense(hdims, hdims * 2; init=normal_init()),
                                Dense(hdims, hdims * 2; init=normal_init())),
                       Dense(hdims * 2, hdims; init=normal_init()))
    aux_model = Chain(Dense(hdims, hdims * 2, relu; init=normal_init()), Dense(hdims * 2, hdims; init=normal_init()))
    args = model_type == "vanilla" ? (main_model,) : (model_type == "skip" ? (main_model, aux_model) : (main_model,))
    _deq = model_type == "vanilla" ? DeepEquilibriumNetwork : SkipDeepEquilibriumNetwork

    return gpu(DEQChain(Dense(1, hdims, relu; init=normal_init()),
                        _deq(args..., get_default_dynamicss_solver(Float32(abstol), Float32(reltol), BS3());
                             sensealg=get_default_ssadjoint(Float32(abstol), Float32(reltol), 50), maxiters=50,
                             verbose=false, jacobian_regularization=jacobian_regularization,
                             residual_regularization=residual_regularization), Dense(hdims, 1)))
end

## Utilities
register_nfe_counts(deq, buffer) = () -> push!(buffer, get_and_clear_nfe!(deq))

h(x) = 3 * (x^3) / 2 + x^2 - 5 * x + 2 * sin(x) - 3

## Training Function
function train(config::Dict, name_extension::String="")
    display(config)

    ## Setup Logging & Experiment Configuration
    expt_name = "toy-$(now())-$(name_extension)"
    lg_wandb = WandbLogger(; project="FastDEQ.jl", name=expt_name, config=config)
    lg_term = PrettyTableLogger("logs/" * expt_name * ".csv", ["Epoch Number", "Train/Time"],
                                ["Train/Running/NFE", "Train/Running/Loss"])


    ## Data Generation
    Random.seed!(0)
    batch_size = get_config(lg_wandb, "batch_size")
    x_data = gpu(rand(Float32, 1, get_config(lg_wandb, "data_size")) .* 2 .- 1)
    y_data = gpu(h.(x_data))

    ## Reproducibility
    Random.seed!(get_config(lg_wandb, "seed"))
    batch_idxs = Iterators.partition(1:size(x_data, 2), batch_size)
    x_data_partition = (x_data[:, i] for i in batch_idxs)
    y_data_partition = (y_data[:, i] for i in batch_idxs)

    ## Model Setup
    model = get_model(get_config(lg_wandb, "hidden_dims"), get_config(lg_wandb, "abstol"),
                      get_config(lg_wandb, "reltol"), get_config(lg_wandb, "model_type"),
                      get_config(lg_wandb, "jacobian_regularization"), get_config(lg_wandb, "residual_regularization"))

    model_type = get_config(lg_wandb, "model_type")
    jac_reg = get_config(lg_wandb, "jacobian_regularization")
    loss_function = SupervisedLossContainer(Flux.Losses.mse, model_type == "vanilla" ? 0.0f0 : 1.0f-2,
                                            jac_reg ? 1.0f0 : 0.0f0, 1.0f-3)

    nfe_counts = Int64[]
    cb = register_nfe_counts(model, nfe_counts)

    ## Training Loop
    ps = Flux.params(model)
    opt = ADAM(get_config(lg_wandb, "learning_rate"))
    step = 1
    for epoch in 1:get_config(lg_wandb, "epochs")
        try
            epoch_loss = 0.0f0
            epoch_nfe = 0
            epoch_time = 0
            for (x, y) in zip(x_data_partition, y_data_partition)
                start_time = time()
                loss, back = Zygote.pullback(() -> loss_function(model, x, y), ps)
                gs = back(one(loss))
                Flux.Optimise.update!(opt, ps, gs)
                epoch_time += time() - start_time

                ### Clear the NFE Count
                get_and_clear_nfe!(model)

                ### Log the losses
                loss = loss_function.loss_function(first(model(x)), y)
                cb()

                log(lg_wandb,
                    Dict("Training/Step/Loss" => loss, "Training/Step/NFE" => nfe_counts[end],
                         "Training/Step/Count" => step))
                lg_term(; records=Dict("Train/Running/NFE" => nfe_counts[end], "Train/Running/Loss" => loss))
                step += 1
                epoch_nfe += nfe_counts[end] * size(x, ndims(x))
                epoch_loss += loss * size(x, ndims(x))
            end
            ### Log the epoch loss
            epoch_loss /= size(x_data, ndims(x_data))
            epoch_nfe /= size(x_data, ndims(x_data))
            log(lg_wandb,
                Dict("Training/Epoch/Loss" => epoch_loss, "Training/Epoch/NFE" => epoch_nfe,
                     "Training/Epoch/Count" => epoch))
            lg_term(epoch, epoch_time)
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

    return model, nfe_counts, x_data, y_data
end

## Run the experiment
experimental_configurations = []
for seed in [11, 111]
    for model_type in ["skipnoextraparams", "vanilla", "skip"]
        for jacobian_regularization in [false, true]
            for residual_regularization in [false, true]
                push!(experimental_configurations,
                      (seed=seed, mtype=model_type, jacreg=jacobian_regularization, resreg=residual_regularization))
            end
        end
    end
end

for expt in experimental_configurations
    name = join(map((k, v) -> "$k-$v", keys(expt), values(expt)), "_")
    @show name

    config = Dict("seed" => expt.seed, "learning_rate" => 1.0f-3, "abstol" => 1.0f-3, "reltol" => 1.0f-3,
                  "epochs" => 300, "batch_size" => 128, "data_size" => 512, "hidden_dims" => 50,
                  "model_type" => expt.mtype, "jacobian_regularization" => expt.jacreg,
                  "residual_regularization" => expt.resreg)

    train(config, name)
end
