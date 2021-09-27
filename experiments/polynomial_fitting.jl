# Comparing SkipDEQ and DEQ on a Polynomial Fitting Problem

## Load Packages
using CUDA,
    Dates,
    FastDEQ,
    Flux,
    OrdinaryDiffEq,
    SteadyStateDiffEq,
    Plots,
    Random,
    Wandb,
    Zygote

CUDA.allowscalar(false)

## Model Defination
struct BranchAndMerge{B1,B2,F}
    branch_1::B1
    branch_2::B2
    final::F
end

Flux.@functor BranchAndMerge

(bam::BranchAndMerge)(x, y) =
    bam.final(tanh.(bam.branch_1(x) .+ bam.branch_2(y)))

## Model and Loss Function
loss_function(x, y, model::DeepEquilibriumNetwork) = mean(abs2, model(x) .- y)

function loss_function(x, y, model::SkipDeepEquilibriumNetwork; λ = 1f-2)
    ŷ, s = model(x)
    return mean(abs2, ŷ .- y) + λ * mean(abs2, ŷ .- s)
end

function get_model(
    hdims::Int,
    abstol::T,
    reltol::T,
    model_type::String,
) where {T}
    if model_type == "vanilla"
        model =
            DeepEquilibriumNetwork(
                BranchAndMerge(
                    Dense(1, hdims),
                    Dense(1, hdims),
                    Dense(hdims, 1),
                ),
                DynamicSS(Tsit5(); abstol = abstol, reltol = reltol),
            ) |> gpu
    elseif model_type == "skip"
        model =
            SkipDeepEquilibriumNetwork(
                BranchAndMerge(
                    Dense(1, hdims),
                    Dense(1, hdims),
                    Dense(hdims, 1),
                ),
                Chain(Dense(1, hdims ÷ 4, relu), Dense(hdims ÷ 4, 1)),
                DynamicSS(Tsit5(); abstol = abstol, reltol = reltol),
            ) |> gpu
    else
        throw(ArgumentError("$model_type must be either `vanilla` or `skip`"))
    end
    return model
end

## Training Function
function train(config::Dict)
    ## Setup Logging & Experiment Configuration
    lg = WandbLogger(
        project = "FastDEQ.jl",
        name = "fastdeqjl-polynomial_fitting-$(now())",
        config = config,
    )

    ## Reproducibility
    Random.seed!(get_config(lg, "seed"))

    ## Data Generation
    batch_size = get_config(lg, "batch_size")
    x_data = rand(Float32, 1, get_config(lg, "data_size")) .* 2 .- 1 |> gpu
    y_data = x_data .^ 2 |> gpu

    x_data_partition = (
        x_data[:, (i-1)*batch_size+1:i*batch_size] for
        i = 1:size(x_data, 2)÷batch_size
    )
    y_data_partition = (
        y_data[:, (i-1)*batch_size+1:i*batch_size] for
        i = 1:size(y_data, 2)÷batch_size
    )

    ## Model Setup
    model = get_model(
        get_config(lg, "hdims"),
        get_config(lg, "abstol"),
        get_config(lg, "reltol"),
        get_config(lg, "model_type"),
    )

    nfe_counts = Int64[]
    cb = register_nfe_counts(model, nfe_counts)

    ## Training Loop
    ps = Flux.params(model)
    opt = ADAM(get_config(lg, "learning_rate"))
    step = 1
    for epoch = 1:get_config(lg, "epochs")
        try
            epoch_loss = 0.0f0
            epoch_nfe = 0
            for (x, y) in zip(x_data_partition, y_data_partition)
                loss, back = Zygote.pullback(() -> loss_function(x, y, model), ps)
                gs = back(one(loss))
                Flux.Optimise.update!(opt, ps, gs)

                ### Store the NFE Count
                cb()

                ### Log the model parameters
                log(lg, cpu; parameters = ps, gradients = gs, commit = false)

                ### Log the losses
                log(
                    lg,
                    Dict(
                        "Training/Step/Loss" => loss,
                        "Training/Step/NFE" => nfe_counts[end],
                        "Training/Step/Count" => step,
                    ),
                )
                step += 1
                epoch_nfe += nfe_counts[end] * size(x, ndims(x))
                epoch_loss += loss * size(x, ndims(x))
            end
            ### Log the epoch loss
            epoch_loss /= size(x_data, ndims(x_data))
            epoch_nfe /= size(x_data, ndims(x_data))
            log(
                lg,
                Dict(
                    "Training/Epoch/Loss" => epoch_loss,
                    "Training/Epoch/NFE" => epoch_nfe,
                    "Training/Epoch/Count" => epoch,
                ),
            )
        catch ex
            if ex isa StopException
                break
            elseif ex isa SkipException
                continue
            else
                rethrow(ex)
            end
        end
    end

    close(lg)

    return model, nfe_counts, x_data, y_data
end


## Run Experiment
config = Dict("seed" => 1,
              "learning_rate" => 0.001,
              "abstol" => 1f-3,
              "reltol" => 1f-3,
              "epochs" => 1000,
              "batch_size" => 64,
              "data_size" => 512,
              "hidden_dims" => 100,
              "model_type" => "vanilla")

model, nfe_counts, x_data, y_data = train(config)



