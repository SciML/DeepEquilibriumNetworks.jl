using FastDEQ, Statistics, Dates, Random
using FastDEQ: get_materials_project_dataloaders

CUDA.allowscalar(false)
enable_fast_mode!()

function loss_function(model, atom_features, neighbor_features, neighbor_feature_indices, crystal_atom_indices,
                       target_standardized)
    prediction = model(atom_features, neighbor_features, neighbor_feature_indices, crystal_atom_indices)
    return mean(abs2, prediction .- target_standardized)
end

function train_one_epoch!(model, dataloader, ps, opt, data_μ, data_σ)
    training_time = actual_loss = total_data = nfe_count = 0

    for (atom_features, neighbor_features, neighbor_feature_indices, crystal_atom_indices, target) in dataloader
        atom_features = gpu(atom_features)
        neighbor_features = gpu(neighbor_features)
        target = gpu(target)
        target_standardized = (target .- data_μ) ./ data_σ

        start_time = time()
        res = Zygote.withgradient(() -> loss_function(model, atom_features, neighbor_features, neighbor_feature_indices,
                                                      crystal_atom_indices, target_standardized), ps)
        Flux.Optimise.update!(opt, ps, res.grad)
        training_time += time() - start_time

        pred = model(atom_features, neighbor_features, neighbor_feature_indices, crystal_atom_indices)

        nfe_count += get_and_clear_nfe!(model) * size(atom_features, 2)
        actual_loss += mean(abs, pred .* data_σ .+ data_μ .- target) * size(atom_features, 2)
        total_data += size(atom_features, 2)
    end

    return (nfe=nfe_count / total_data, training_time=training_time, loss=actual_loss / total_data,
            training_time_per_batch=training_time / length(dataloader))
end

function compute_mae(model, dataloader, data_μ, data_σ)
    mae = total_elements = pred_time = nfe_count = 0

    for (atom_features, neighbor_features, neighbor_feature_indices, crystal_atom_indices, target) in dataloader
        atom_features = gpu(atom_features)
        neighbor_features = gpu(neighbor_features)
        target = gpu(target)

        start_time = time()
        mae += mean(abs,
                    (model(atom_features, neighbor_features, neighbor_feature_indices, crystal_atom_indices) .*
                     data_σ .+ data_μ) .- target) * size(atom_features, ndims(atom_features))
        pred_time += time() - start_time

        nfe_count += get_and_clear_nfe!(model) * size(atom_features, ndims(atom_features))
        total_elements += size(atom_features, ndims(atom_features))
    end

    return (nfe=nfe_count / total_elements, loss=mae / total_elements, prediction_time=pred_time,
            prediction_time_per_batch=pred_time / length(dataloader))
end

function train(name_extension::String=""; epochs, start_learning_rate, weight_decay, seed, root_dir, batchsize,
               eval_batchsize)
    expt_name = "fastdeqjl-supervised_mnist_classification-$(now())-$(name_extension)"
    lg_term = PrettyTableLogger("logs/" * expt_name * ".csv",
                                ["Epoch", "Train/NFE  ", "Train/Loss", "Train/Time", "Train/Time Per Batch",
                                 "Validation/NFE  ", "Validation/Loss", "Validation/Time", "Validation/Time Per Batch",
                                 "Test/NFE  ", "Test/Loss", "Test/Time", "Test/Time Per Batch"])

    ## Reproducibility
    Random.seed!(seed)

    ## Data Loaders
    train_dataloader, val_dataloader, test_dataloader = get_materials_project_dataloaders(root_dir, batchsize,
                                                                                          eval_batchsize; verbose=false,
                                                                                          seed=seed, train_split=0.8,
                                                                                          val_split=0.1)

    _targets = hcat(last.(train_dataloader)...)
    data_μ = Float32(mean(_targets))
    data_σ = Float32(std(_targets))

    ## Model
    model = gpu(MaterialsProjectCrystalGraphConvNet(; original_atom_feature_length=92, neighbor_feature_length=41))

    ### Some weird CUDNN error on cyclops for inference mode
    for c in model.convs
        c.bn1.active = true
        c.bn2.active = true
    end

    ## Optimizer
    ps = Flux.params(model)
    opt = ADAMW(start_learning_rate, (0.9, 0.999), weight_decay)

    ## Training
    for e in 1:epochs
        train_stats = train_one_epoch!(model, train_dataloader, ps, opt, data_μ, data_σ)
        val_stats = compute_mae(model, val_dataloader, data_μ, data_σ)
        test_stats = compute_mae(model, test_dataloader, data_μ, data_σ)

        lg_term(e, train_stats.nfe, train_stats.loss, train_stats.training_time, train_stats.training_time_per_batch,
                val_stats.nfe, val_stats.loss, val_stats.prediction_time, val_stats.prediction_time_per_batch,
                test_stats.nfe, test_stats.loss, test_stats.prediction_time, test_stats.prediction_time_per_batch)

        ## Reduce LR after 10 epochs
        e == 10 && (opt[3].eta = opt[3].eta / 10)
    end

    close(lg_term)

    return nothing
end

train(; epochs=25, start_learning_rate=1e-2, weight_decay=1e-5, seed=0, root_dir="data/mp/3402", batchsize=256,
      eval_batchsize=256)
