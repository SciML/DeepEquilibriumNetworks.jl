# Comparing SkipDEQ and DEQ on a Polynomial Fitting Problem

## Load Packages
using CUDA,
    Dates,
    DiffEqSensitivity,
    FastDEQ,
    Flux,
    OrdinaryDiffEq,
    Statistics,
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
    bam.final(gelu.(bam.branch_1(x) .+ bam.branch_2(y)))

## Model and Loss Function
function get_model(
    hdims::Int,
    abstol::T,
    reltol::T,
    model_type::String,
) where {T}
    if model_type == "vanilla"
        model =
            DEQChain(
                Dense(1, hdims, gelu),
                DeepEquilibriumNetwork(
                    BranchAndMerge(
                        Dense(hdims, hdims * 2),
                        Dense(hdims, hdims * 2),
                        Dense(hdims * 2, hdims),
                    ),
                    DynamicSS(RK4(); abstol = abstol, reltol = reltol),
                    sensealg = SteadyStateAdjoint(
                        autodiff = true,
                        autojacvec = ZygoteVJP(),
                        linsolve = LinSolveKrylovJL(),
                    ),
                    maxiters = 50,
                ),
                Dense(hdims, 1),
            ) |> gpu
    elseif model_type == "skip"
        model =
            DEQChain(
                Dense(1, hdims, gelu),
                SkipDeepEquilibriumNetwork(
                    BranchAndMerge(
                        Dense(hdims, hdims * 2),
                        Dense(hdims, hdims * 2),
                        Dense(hdims * 2, hdims),
                    ),
                    Chain(
                        Dense(hdims, hdims * 5, gelu),
                        Dense(hdims * 5, hdims),
                    ),
                    DynamicSS(Tsit5(); abstol = abstol, reltol = reltol),
                    sensealg = SteadyStateAdjoint(
                        autodiff = true,
                        autojacvec = ZygoteVJP(),
                        linsolve = LinSolveKrylovJL(),
                    ),
                    maxiters = 50,
                ),
                Dense(hdims, 1),
            ) |> gpu
    else
        throw(ArgumentError("$model_type must be either `vanilla` or `skip`"))
    end
    return model
end

## Utilities
function register_nfe_counts(deq, buffer)
    callback() = push!(buffer, get_and_clear_nfe!(deq))
    return callback
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
        get_config(lg, "hidden_dims"),
        get_config(lg, "abstol"),
        get_config(lg, "reltol"),
        get_config(lg, "model_type"),
    )

    loss_function = SupervisedLossContainer(
        (ŷ, y) -> mean(abs2, ŷ .- y),
        1.0f-2
    )

    nfe_counts = Int64[]
    cb = register_nfe_counts(model, nfe_counts)

    ## Training Loop
    ps = Flux.params(model)
    opt = ADAMW(get_config(lg, "learning_rate"), (0.9, 0.999), 0.001)
    step = 1
    for epoch = 1:get_config(lg, "epochs")
        try
            epoch_loss = 0.0f0
            epoch_nfe = 0
            for (x, y) in zip(x_data_partition, y_data_partition)
                loss, back =
                    Zygote.pullback(() -> loss_function(model, x, y), ps)
                gs = back(one(loss))
                Flux.Optimise.update!(opt, ps, gs)

                ### Clear the NFE Count
                get_and_clear_nfe!(model)

                ### Log the losses
                ŷ = model(x)
                loss = loss_function.loss_function(ŷ isa Tuple ? ŷ[1] : ŷ, y)
                cb()

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

## Plotting
function plot_nfe_counts(nfe_counts_1, nfe_counts_2)
    p = plot(nfe_counts_1, label = "Vanilla DEQ")
    plot!(p, nfe_counts_2, label = "Skip DEQ")
    xlabel!(p, "Training Iteration")
    ylabel!(p, "NFE Count")
    title!(p, "NFE over Training Iterations of DEQ vs SkipDEQ")
    return p
end

## Run Experiment
nfe_count_dict = Dict("vanilla" => [], "skip" => [])

for seed in [1, 11, 111]
    for model_type in ["skip", "vanilla"]
        config = Dict(
            "seed" => seed,
            "learning_rate" => 1f-3,
            "abstol" => 1f-3,
            "reltol" => 1f-3,
            "epochs" => 250,
            "batch_size" => 128,
            "data_size" => 512,
            "hidden_dims" => 50,
            "model_type" => model_type,
        )

        model, nfe_counts, x_data, y_data = train(config)

        push!(nfe_count_dict[model_type], nfe_counts)
    end
end

plot_nfe_counts(
    vec(mean(hcat(nfe_count_dict["vanilla"]...), dims = 2)),
    vec(mean(hcat(nfe_count_dict["skip"]...), dims = 2)),
)