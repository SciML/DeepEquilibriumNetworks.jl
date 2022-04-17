abstract type AbstractTaskModelConfiguration end

# Predefined Image Classification Models
Base.@kwdef struct ImageClassificationModelConfiguration{N} <: AbstractTaskModelConfiguration
    num_layers::Int
    num_classes::Int
    dropout_rate::Float32
    group_count::Int
    weight_norm::Bool
    downsample_times::Int
    expansion_factor::Int
    post_gn_affine::Bool
    image_size::Tuple{Int,Int}

    num_modules::Int
    num_branches::Int
    block_type::Symbol
    big_kernels::NTuple{N,Int}
    head_channels::NTuple{N,Int}
    num_blocks::NTuple{N,Int}
    num_channels::NTuple{N,Int}

    fuse_method::Symbol
    final_channelsize::Int

    fwd_maxiters::Int
    bwd_maxiters::Int
    model_type::Symbol
    continuous::Bool

    # Specific for Continuous Models
    abstol::Float32 = 5f-2
    reltol::Float32 = 5f-2
    stop_mode::Symbol = :rel_norm
    ode_solver = VCABM3()
end

function get_model_config(dataset::Symbol, model_size::Symbol; kwargs...)
    if dataset == :CIFAR10
        if model_size == :TINY
            return ImageClassificationModelConfiguration{2}(;
                num_layers=10,
                num_classes=10,
                dropout_rate=0.25f0,
                group_count=8,
                weight_norm=true,
                downsample_times=0,
                expansion_factor=5,
                post_gn_affine=false,
                image_size=(32, 32),
                num_modules=1,
                num_branches=2,
                block_type=:basic,
                big_kernels=(0, 0),
                head_channels=(8, 16),
                num_blocks=(1, 1),
                num_channels=(24, 24),
                fuse_method=:sum,
                final_channelsize=200,
                fwd_maxiters=18,
                bwd_maxiters=20,
                kwargs...
            )
        elseif model_size == :LARGE
            return ImageClassificationModelConfiguration{4}(;
                num_layers=10,
                num_classes=10,
                dropout_rate=0.3f0,
                group_count=8,
                weight_norm=true,
                downsample_times=0,
                expansion_factor=5,
                post_gn_affine=false,
                image_size=(32, 32),
                num_modules=1,
                num_branches=4,
                block_type=:basic,
                big_kernels=(0, 0, 0, 0),
                head_channels=(14, 28, 56, 112),
                num_blocks=(1, 1, 1, 1),
                num_channels=(32, 64, 128, 256),
                fuse_method=:sum,
                final_channelsize=1680,
                fwd_maxiters=18,
                bwd_maxiters=20,
                kwargs...
            )
        else
            throw(ArgumentError("`model_size` must be one of `[:TINY, :LARGE]`"))
        end
    elseif dataset == :IMAGENET
        if model_size == :SMALL
            return ImageClassificationModelConfiguration{4}(;
                num_layers=4,
                num_classes=1000,
                dropout_rate=0.0f0,
                group_count=8,
                weight_norm=true,
                downsample_times=2,
                expansion_factor=5,
                post_gn_affine=true,
                image_size=(224, 224),
                num_modules=1,
                num_branches=4,
                block_type=:basic,
                big_kernels=(0, 0, 0, 0),
                head_channels=(24, 48, 96, 192),
                num_blocks=(1, 1, 1, 1),
                num_channels=(32, 64, 128, 256),
                fuse_method=:sum,
                final_channelsize=2048,
                fwd_maxiters=27,
                bwd_maxiters=28,
                kwargs...
            )
        elseif model_size == :LARGE
            return ImageClassificationModelConfiguration{4}(;
                num_layers=4,
                num_classes=1000,
                dropout_rate=0.0f0,
                group_count=8,
                weight_norm=true,
                downsample_times=2,
                expansion_factor=5,
                post_gn_affine=true,
                image_size=(224, 224),
                num_modules=1,
                num_branches=4,
                block_type=:basic,
                big_kernels=(0, 0, 0, 0),
                head_channels=(32, 64, 128, 256),
                num_blocks=(1, 1, 1, 1),
                num_channels=(80, 160, 320, 640),
                fuse_method=:sum,
                final_channelsize=2048,
                fwd_maxiters=27,
                bwd_maxiters=28,
                kwargs...
            )
        elseif model_size == :XL
            return ImageClassificationModelConfiguration{4}(;
                num_layers=4,
                num_classes=1000,
                dropout_rate=0.0f0,
                group_count=8,
                weight_norm=true,
                downsample_times=2,
                expansion_factor=5,
                post_gn_affine=true,
                image_size=(224, 224),
                num_modules=1,
                num_branches=4,
                block_type=:basic,
                big_kernels=(0, 0, 0, 0),
                head_channels=(32, 64, 128, 256),
                num_blocks=(1, 1, 1, 1),
                num_channels=(88, 176, 352, 704),
                fuse_method=:sum,
                final_channelsize=2048,
                fwd_maxiters=27,
                bwd_maxiters=28,
                kwargs...
            )
        else
            throw(ArgumentError("`model_size` must be one of `[:SMALL, :LARGE, :XL]`"))
        end
    else
        throw(ArgumentError("`dataset` must be one of `[:CIFAR10]`"))
    end
end

function compute_feature_scales(config::ImageClassificationModelConfiguration)
    image_size = config.image_size
    image_size_downsampled = image_size
    for _ in 1:(config.downsample_times)
        image_size_downsampled = image_size_downsampled .÷ 2
    end
    scales = [(image_size_downsampled..., config.num_channels[1])]
    for i in 2:(config.num_branches)
        push!(scales, ((scales[end][1:2] .÷ 2)..., config.num_channels[i]))
    end
    return Tuple(scales)
end

# Experiment Configuration
Base.@kwdef struct ExperimentConfiguration{M<:AbstractTaskModelConfiguration}
    model_config::M

    # Eval
    eval_batchsize::Int

    # Train
    train_batchsize::Int
    nepochs::Int
    pretrain_steps::Int

    # Optimiser
    lr_scheduler::Symbol
    optimiser::Symbol
    eta::Float32
    momentum::Float32
    nesterov::Bool
    weight_decay::Float32
end

function get_experiment_config(dataset::Symbol, model_size::Symbol; kwargs...)
    if dataset == :CIFAR10
        if model_size == :TINY
            return ExperimentConfiguration(
                model_config=get_model_config(dataset, model_size; kwargs...),
                eval_batchsize=64,
                train_batchsize=64,
                nepochs=50,
                pretrain_steps=0 ÷ (is_distributed() ? total_workers() : 1),
                lr_scheduler=:COSINE,
                optimiser=:ADAM,
                eta=0.001f0 / 2 * (is_distributed() ? total_workers() : 1),
                weight_decay=0.0f0,
                momentum=0.9f0,
                nesterov=true
            )
        elseif model_size == :LARGE
            return ExperimentConfiguration(
                model_config=get_model_config(dataset, model_size; kwargs...),
                eval_batchsize=32,
                train_batchsize=32,
                nepochs=220,
                pretrain_steps=20000 ÷ (is_distributed() ? total_workers() : 1),
                lr_scheduler=:COSINE,
                optimiser=:ADAM,
                eta=0.001f0 / 4 * (is_distributed() ? total_workers() : 1),
                weight_decay=0.0f0,
                momentum=0.9f0,
                nesterov=true
            )
        else
            throw(ArgumentError("`model_size` must be one of `[:TINY, :LARGE]`"))
        end
    elseif dataset == :IMAGENET
        if model_size == :SMALL
            return ExperimentConfiguration(
                model_config=get_model_config(dataset, model_size; kwargs...),
                eval_batchsize=32,
                train_batchsize=32,
                nepochs=100,
                pretrain_steps=510000 ÷ (is_distributed() ? total_workers() : 1),
                lr_scheduler=:COSINE,
                optimiser=:SGD,
                eta=0.05f0 / 4 * (is_distributed() ? total_workers() : 1),
                weight_decay=0.00005f0,
                momentum=0.9f0,
                nesterov=true
            )
        elseif model_size == :LARGE
            return ExperimentConfiguration(
                model_config=get_model_config(dataset, model_size; kwargs...),
                eval_batchsize=32,
                train_batchsize=32,
                nepochs=100,
                pretrain_steps=510000 ÷ (is_distributed() ? total_workers() : 1),
                lr_scheduler=:COSINE,
                optimiser=:SGD,
                eta=0.05f0 / 4 * (is_distributed() ? total_workers() : 1),
                weight_decay=0.00005f0,
                momentum=0.9f0,
                nesterov=true
            )
        elseif model_size == :XL
            return ExperimentConfiguration(
                model_config=get_model_config(dataset, model_size; kwargs...),
                eval_batchsize=32,
                train_batchsize=32,
                nepochs=100,
                pretrain_steps=510000 ÷ (is_distributed() ? total_workers() : 1),
                lr_scheduler=:COSINE,
                optimiser=:SGD,
                eta=0.05f0 / 8 * (is_distributed() ? total_workers() : 1),
                weight_decay=0.00005f0,
                momentum=0.9f0,
                nesterov=true
            )
        else
            throw(ArgumentError("`model_size` must be one of `[:SMALL, :LARGE, :XL]`"))
        end
    else
        throw(ArgumentError("`dataset` must be one of `[:CIFAR10]`"))
    end
end

