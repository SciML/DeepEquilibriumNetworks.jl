using FastDEQExperiments, Flux, CUDA, Optimisers, Dates, FluxMPI

# Distributed Training
FluxMPI.Init(; verbose=true)

# Setup
CUDA.allowscalar(false)

# Training
function train_model(config, expt_name)
    # Logger Setup
    mkpath("logs/")
    lg = FastDEQExperiments.PrettyTableLogger(
        joinpath("logs/", expt_name * ".csv"),
        ["Epoch Number", "Train/Time", "Test/NFE", "Test/Accuracy", "Test/Loss", "Test/Time"],
        ["Train/Running/NFE", "Train/Running/Loss", "Train/Running/Accuracy"],
    )

    # Experiment Configuration
    expt_config = FastDEQExperiments.get_experiment_config(
        :CIFAR10,
        config["model_size"];
        model_type=config["model_type"],
        continuous=config["continuous"],
        abstol=config["abstol"],
        reltol=config["reltol"],
    )

    # Model Setup
    model, ps, st = FastDEQExperiments.get_model(expt_config; seed=config["seed"], device=gpu, warmup=true)

    # Get Dataloaders
    train_dataloader, test_dataloader = FastDEQExperiments.get_dataloaders(
        :CIFAR10; train_batchsize=expt_config.train_batchsize, eval_batchsize=expt_config.eval_batchsize
    )

    # Train
    ps, st, st_opt = FastDEQExperiments.train(
        model,
        ps,
        st,
        FastDEQExperiments.loss_function(expt_config),
        FastDEQExperiments.construct_optimiser(expt_config)...,
        train_dataloader,
        nothing,
        test_dataloader,
        gpu,
        expt_config.nepochs,
        lg,
        expt_config,
    )

    # Close Logger and Flush Data to disk
    Base.close(lg)

    return model, cpu(ps), cpu(st), st_opt
end

# Experiment Configurations
configs = []
for seed in [6171, 3859, 2961], model_type in [:VANILLA, :SKIP, :SKIPV2], model_size in [:TINY, :LARGE]
    push!(
        configs,
        Dict(
            "seed" => seed,
            "abstol" => 5.0f-2,
            "reltol" => 5.0f-2,
            "model_type" => model_type,
            "continuous" => true,
            "model_size" => model_size,
        ),
    )
end

# Training
for config in configs
    expt_name = "cifar-10_seed-$(config["seed"])_model-$(config["model_type"])_size-$(config["model_size"])_continuous-$(config["continuous"])_now-$(now())"
    FastDEQExperiments._should_log() && println("Starting Experiment: " * expt_name)
    model, ps, st, st_opt = train_model(config, expt_name)
end
