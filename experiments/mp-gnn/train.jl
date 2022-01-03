using FastDEQ, Statistics

function train_one_epoch!(model, dataloader, ps, opt, data_μ, data_σ)
    data_μ = gpu(data_μ)
    data_σ = gpu(data_σ)
    training_time = 0
    actual_loss = 0
    total_data = 0
    for (atom_features, neighbor_features, neighbor_feature_indices, crystal_atom_indices, target) in dataloader
        atom_features = gpu(atom_features)
        neighbor_features = gpu(neighbor_features)
        neighbor_feature_indices = gpu(neighbor_feature_indices)
        target = gpu(target)
        target_standardized = (target .- data_μ) ./ data_σ

        start_time = time()
        res = Zygote.withgradient(() -> mean(abs2,
                                             model(atom_features, neighbor_features, neighbor_feature_indices,
                                                   crystal_atom_indices) .- target_standardized), ps)
        gs = res.grad
        Flux.Optimise.update!(opt, ps, gs)
        training_time += time() - start_time

        pred = model(atom_features, neighbor_features, neighbor_feature_indices, crystal_atom_indices)

        actual_loss += mean(abs, pred .* data_σ .+ data_μ .- target) * size(atom_features, 2)
        total_data += size(atom_features, 2)
    end

    actual_loss /= total_data

    @show actual_loss, training_time / length(dataloader)

    return (training_time=training_time, loss=actual_loss)
end

function compute_mae(model, dataloader, data_μ, data_σ)
    data_μ = gpu(data_μ)
    data_σ = gpu(data_σ)
    mae = 0
    total_elements = 0
    for (atom_features, neighbor_features, neighbor_feature_indices, crystal_atom_indices, target) in dataloader
        atom_features = gpu(atom_features)
        neighbor_features = gpu(neighbor_features)
        neighbor_feature_indices = gpu(neighbor_feature_indices)
        target = gpu(target)

        mae += mean(abs,
                    (model(atom_features, neighbor_features, neighbor_feature_indices, crystal_atom_indices) .*
                     data_σ .+ data_μ) .- target) * size(atom_features, ndims(atom_features))
        total_elements += size(atom_features, ndims(atom_features))
    end
    return mae / total_elements
end

trd, vd, ted = FastDEQ.get_materials_project_dataloaders("data/mp/46744/", 256, 1024; verbose=true, seed=1,
                                                         train_split=0.8, val_split=0.1)

_targets = hcat(last.(trd)...)
data_μ = Float32(mean(_targets))
data_σ = Float32(std(_targets))

model = gpu(MaterialsProjectCrystalGraphConvNet(; original_atom_feature_length=92, neighbor_feature_length=41))

val_losses = []
test_losses = []

ps = Flux.params(model);
opt = ADAMW(1.0f-2, (0.9, 0.999), 0.0)
for e in 1:1000
    @show e
    train_one_epoch!(model, trd, ps, opt, data_μ, data_σ)
    if e == 1
        opt[3].eta = opt[3].eta / 10
    end
    push!(val_losses, compute_mae(model, vd, data_μ, data_σ))
    push!(test_losses, compute_mae(model, ted, data_μ, data_σ))
    @show val_losses[end], test_losses[end]
end

idx = argmin(val_losses)
@show val_losses[idx], test_losses[idx]

function train_and_validate() end

function run() end
