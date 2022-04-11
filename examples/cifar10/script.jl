# ----------------------------------- #
# ------ Precompilation in HPC ------ #
d = strip(String(read(`mktemp -d`)))
mkdir(joinpath(d, "compiled"))
pushfirst!(DEPOT_PATH, d)
#------------------------------------ #

using FastDEQExperiments, Flux, CUDA, Optimisers, Dates, FluxMPI

# Distributed Training
FluxMPI.Init(; verbose=true)

# Setup
CUDA.versioninfo()
CUDA.math_mode!(CUDA.FAST_MATH)
CUDA.allowscalar(false)

function invoke_gc()
    GC.gc(true)
    CUDA.reclaim()
    return nothing
end

# Hyperparameters
config = Dict(
    "seed" => 0,
    "learning_rate" => 0.001f0,
    "abstol" => 5.0f-2,
    "reltol" => 5.0f-2,
    "maxiters" => 20,
    "epochs" => 50,
    "dropout_rate" => 0.25f0,
    "batchsize" => 64,
    "eval_batchsize" => 64,
    "model_type" => :skip,
    "continuous" => true,
    "weight_decay" => 0.0000025f0,
)

expt_name = "cifar-10_seed-$(config["seed"])_model-$(config["model_type"])_continuous-$(config["continuous"])_now-$(now())"

# Training
function train_model(config, expt_name)
    # Logger Setup
    mkpath("logs/")
    lg = FastDEQExperiments.PrettyTableLogger(
        joinpath("logs/", expt_name * ".csv"),
        ["Epoch Number", "Train/Time", "Test/NFE", "Test/Accuracy", "Test/Loss", "Test/Time"],
        ["Train/Running/NFE", "Train/Running/Loss", "Train/Running/Accuracy"],
    )

    # Model Setup
    model, ps, st = FastDEQExperiments.get_model(
        Val(:CIFAR10);
        dropout_rate=config["dropout_rate"],
        model_type=config["model_type"],
        continuous=config["continuous"],
        maxiters=config["maxiters"],
        abstol=config["abstol"],
        reltol=config["reltol"],
        seed=config["seed"],
        device=gpu,
        warmup=true,
        group_count=8,
    )

    # Get Dataloaders
    train_dataloader, test_dataloader = FastDEQExperiments.get_dataloaders(
        :CIFAR10; train_batchsize=config["batchsize"], eval_batchsize=config["eval_batchsize"]
    )

    # Train
    ps, st, st_opt = FastDEQExperiments.train(
        model,
        ps,
        st,
        FastDEQExperiments.loss_function(:CIFAR10, config["model_type"]),
        Optimisers.ADAM(config["learning_rate"]),
        train_dataloader,
        nothing,
        test_dataloader,
        gpu,
        config["epochs"],
        lg;
        cleanup_function=invoke_gc,
    )

    # Close Logger and Flush Data to disk
    Base.close(lg)

    return model, cpu(ps), cpu(st), st_opt
end

model, ps, st, st_opt = train_model(config, expt_name)
