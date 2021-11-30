# Comparing SkipDEQ and DEQ on a Polynomial Fitting Problem

## Load Packages
using CUDA,
    Dates,
    DiffEqSensitivity,
    FastDEQ,
    Flux,
    GLMakie,
    OrdinaryDiffEq,
    Statistics,
    SteadyStateDiffEq,
    Plots,
    Random,
    Zygote

CUDA.allowscalar(false)

## Model and Loss Function
struct PolyFit{T}
    W₁::T
    W₂::T
    U::T
    b::T
end

Flux.@functor PolyFit

PolyFit(dims::Int, hdims::Int) = PolyFit(
    randn(Float32, hdims, dims),
    randn(Float32, dims, hdims),
    randn(Float32, hdims, dims),
    zeros(Float32, hdims, 1),
)

(p::PolyFit)(z, x) = p.W₂ * tanh.(p.W₁ * z .+ p.U * x .+ p.b)

function get_model(
    hdims::Int,
    abstol::T,
    reltol::T,
    model_type::String,
) where {T}
    return (
        model_type == "vanilla" ? DeepEquilibriumNetwork :
        SkipDeepEquilibriumNetwork
    )(
        PolyFit(1, hdims),
        get_default_dynamicss_solver(abstol, reltol),
        sensealg = get_default_ssadjoint(abstol, reltol, 20),
        maxiters = 20,
        verbose = false,
    ) |> gpu
end

## Utilities
function register_nfe_counts(deq, buffer)
    callback() = push!(buffer, get_and_clear_nfe!(deq))
    return callback
end

function generate_polynomial_data(count::Int)
    h(x) = 3 * (x ^ 3) / 2 + x ^ 2 - 5 * x + 2 * sin(x) - 3 + randn() * 0.005
    x = rand(Float32, count) .* 4 .- 2
    return x, h.(x)
end


## Training Function
function train(model_type)
    ## Reproducibility
    Random.seed!(1)

    ## Data Generation
    data_size = 5096
    batch_size = 512
    x_data, y_data = Flux.unsqueeze.(generate_polynomial_data(data_size), 1) .|> gpu

    x_train, y_train = x_data[:, 1:data_size - 1000], y_data[:, 1:data_size - 1000]
    x_val, y_val = x_data[:, data_size - 1000 + 1:end], y_data[:, data_size - 1000 + 1:end]

    x_train_data_partition = (
        x_train[:, (i-1)*batch_size+1:i*batch_size] for
        i = 1:size(x_train, 2)÷batch_size
    )
    y_train_data_partition = (
        y_train[:, (i-1)*batch_size+1:i*batch_size] for
        i = 1:size(y_train, 2)÷batch_size
    )
    x_val_data_partition = (
        x_val[:, (i-1)*batch_size+1:i*batch_size] for
        i = 1:size(x_val, 2)÷batch_size
    )
    y_val_data_partition = (
        y_val[:, (i-1)*batch_size+1:i*batch_size] for
        i = 1:size(y_val, 2)÷batch_size
    )

    ## Model Setup
    model = get_model(50, 1f-2, 1f-2, model_type)

    loss_function = SupervisedLossContainer(Flux.Losses.mse, 5.0f0)

    nfe_counts = Int64[]
    cb = register_nfe_counts(model, nfe_counts)

    ## Training Loop
    ps = Flux.params(model)
    opt = ADAM(0.01)
    for epoch = 1:1000
        try
            epoch_loss_train = 0.0f0
            epoch_nfe_train = 0
            epoch_loss_val = 0.0f0
            epoch_nfe_val = 0
            for (x, y) in zip(x_train_data_partition, y_train_data_partition)
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

                epoch_nfe_train += nfe_counts[end] * size(x, ndims(x))
                epoch_loss_train += loss * size(x, ndims(x))
            end

            for (x, y) in zip(x_val_data_partition, y_val_data_partition)
                ŷ = model(x)
                loss = loss_function.loss_function(ŷ isa Tuple ? ŷ[1] : ŷ, y)

                ### Log the losses
                cb()

                epoch_nfe_val += nfe_counts[end] * size(x, ndims(x))
                epoch_loss_val += loss * size(x, ndims(x))
            end

            ### Log the epoch loss
            epoch_loss_train /= size(x_train, ndims(x_train))
            epoch_nfe_train /= size(x_train, ndims(x_train))
            epoch_loss_val /= size(x_val, ndims(x_val))
            epoch_nfe_val /= size(x_val, ndims(x_val))

            if epoch % 10 == 1
                println("[$model_type] Epoch: $epoch")
                println("\t[Training] Loss: $epoch_loss_train | NFE: $epoch_nfe_train")
                println("\t[Validation] Loss: $epoch_loss_val | NFE: $epoch_nfe_val")
            end
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

    return model, nfe_counts, x_data, y_data
end


## Run Experiments
models = Dict()
datas = Dict()
nfe_counts = Dict()

for model_type in ["skip", "vanilla"]
    model, nfe_count, x_data, y_data = train(model_type)
    models[model_type] = model
    datas[model_type] = (x_data, y_data)
    nfe_counts[model_type] = nfe_count
end

# Generate Trajectories
function generate_trajectory(
    model::DeepEquilibriumNetwork,
    x::AbstractArray,
    depth = 50,
)
    traj = []
    u = zero(x)
    push!(traj, u)
    for _ = 1:(depth-1)
        u = model.re(model.p)(u, x)
        push!(traj, u)
    end
    return traj
end

function generate_trajectory(
    model::SkipDeepEquilibriumNetwork,
    x::AbstractArray,
    depth = 50,
)
    traj = []
    u = model.re1(model.p)(zero(x), x)
    push!(traj, u)
    for _ = 1:(depth-2)
        u = model.re1(model.p)(u, x)
        push!(traj, u)
    end
    return traj
end

traj_vanilla = generate_trajectory(models["vanilla"], reshape(unique(datas["vanilla"][1] |> cpu), (1, :)) |> gpu)
traj_skip = generate_trajectory(models["skip"], reshape(unique(datas["skip"][1] |> cpu), (1, :)) |> gpu)

function plot_trajectories(traj_1, x_data_1, traj_2, x_data_2)
    fig = Figure(resolution = (900, 900))

    ax = Axis3(
        fig,
        aspect = :data,
        perspectiveness = 0.5,
        elevation = π / 9,
        xzpanelcolor = (:black, 0.75),
        yzpanelcolor = (:black, 0.75),
        zgridcolor = :grey,
        ygridcolor = :grey,
        xgridcolor = :grey,
    )

    x_data = vec(x_data_1) |> cpu
    z = vcat(vec.(traj_1)...) |> cpu
    x = repeat(x_data, length(traj_1))
    y =
        Float32.(
            repeat(
                (1:length(traj_1)) ./ length(traj_1),
                inner = length(x_data),
            ),
        )

    GLMakie.surface!(
        ax,
        x,
        y,
        z,
        colormap = :viridis,
        colorrange = (minimum(z), maximum(z)),
    )
    fig[1, 1] = ax

    ax = Axis3(
        fig,
        aspect = :data,
        perspectiveness = 0.5,
        elevation = π / 9,
        xzpanelcolor = (:black, 0.75),
        yzpanelcolor = (:black, 0.75),
        zgridcolor = :grey,
        ygridcolor = :grey,
        xgridcolor = :grey,
    )

    x_data = vec(x_data_2) |> cpu
    z = vcat(vec.(traj_2)...) |> cpu
    x = repeat(x_data, length(traj_2))
    y =
        Float32.(
            repeat(
                (1:length(traj_2)) ./ length(traj_2),
                inner = length(x_data),
            ),
        )

    GLMakie.surface!(
        ax,
        x,
        y,
        z,
        colormap = :viridis,
        colorrange = (minimum(z), maximum(z)),
    )
    fig[1, 2] = ax

    return fig
end

plot_trajectories(
    traj_vanilla,
    reshape(unique(datas["vanilla"][1] |> cpu), (1, :)) |> gpu,
    traj_skip,
    reshape(unique(datas["skip"][1] |> cpu), (1, :)) |> gpu,
)
