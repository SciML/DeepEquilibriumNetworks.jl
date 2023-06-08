import SimpleConfig, OrdinaryDiffEq
import SimpleConfig: @option

@option struct SensitivityConfig
    jfb::Bool = false
    abstol::Float32 = 5.0f-2
    reltol::Float32 = 5.0f-2
    maxiters::Int = 20
end

@option struct SolverConfig
    continuous::Bool = true
    stop_mode::String = "rel_norm"
    ode_solver::String = "vcab3"
    abstol::Float32 = 5.0f-2
    reltol::Float32 = 5.0f-2
    abstol_termination::Float32 = 5.0f-2
    reltol_termination::Float32 = 5.0f-2
end

@option struct ModelConfig
    num_classes::Int = 10
    dropout_rate::Float32 = 0.25f0
    group_count::Int = 8
    weight_norm::Bool = true
    downsample_times::Int = 0
    expansion_factor::Int = 5
    image_size::Vector{Int64} = [32, 32]
    num_branches::Int = 2
    big_kernels::Vector{Int64} = [0, 0]
    head_channels::Vector{Int64} = [8, 16]
    num_channels::Vector{Int64} = [24, 24]
    fuse_method::String = "sum"
    final_channelsize::Int = 200
    model_type::String = "vanilla"
    maxiters::Int = 18
    in_channels::Int = 3
    sensealg::SensitivityConfig = SensitivityConfig()
    solver::SolverConfig = SolverConfig()
end

@option struct OptimizerConfig
    lr_scheduler::String = "cosine"
    optimizer::String = "adam"
    learning_rate::Float32 = 0.01f0
    nesterov::Bool = false
    momentum::Float32 = 0.0f0
    weight_decay::Float32 = 0.0f0
    cycle_length::Int = 50000
    lr_step::Vector{Int64} = [30, 60, 90]
    lr_step_decay::Float32 = 0.1f0
end

@option struct TrainConfig
    total_steps::Int = 50000
    pretrain_steps::Int = 5000
    evaluate_every::Int = 2500
    resume::String = ""
    evaluate::Bool = false
    checkpoint_dir::String = "checkpoints"
    log_dir::String = "logs"
    expt_subdir::String = ""
    expt_id::String = ""
    print_frequency::Int = 100
    w_skip::Float32 = 1.0f0
end

@option struct DatasetConfig
    augment::Bool = false
    data_root::String = ""
    eval_batchsize::Int = 64
    train_batchsize::Int = 64
end

@option struct ExperimentConfig
    seed::Int = 12345
    model::ModelConfig = ModelConfig()
    optimizer::OptimizerConfig = OptimizerConfig()
    train::TrainConfig = TrainConfig()
    dataset::DatasetConfig = DatasetConfig()
end

function _get_ode_solver(cfg::SolverConfig)
    if cfg.ode_solver == "vcabm3"
        return OrdinaryDiffEq.VCABM3()
    elseif cfg.ode_solver == "vcab3"
        return OrdinaryDiffEq.VCAB3()
    elseif cfg.ode_solver == "tsit5"
        return OrdinaryDiffEq.Tsit5()
    else
        throw(ArgumentError("unknown ODE Solver: $(cfg.ode_solver). Supported values are: " *
                            "`vcabm3`, `vcab3`, and `tsit5`"))
    end
end
