function train_one_epoch!(model, dataloader, ps, opt, data_μ, data_σ)
    data_μ = gpu(data_μ)
    data_σ = gpu(data_σ)
    training_time = 0
    for (atom_features, neighbor_features, neighbor_feature_indices, crystal_atom_indices, target) in dataloader
        atom_features = gpu(atom_features)
        neighbor_features = gpu(neighbor_features)
        neighbor_feature_indices = gpu(neighbor_feature_indices)
        target = (gpu(target) .- data_μ) ./ data_σ

        start_time = time()
        res = Zygote.withgradient(() -> mean(abs2,
                                             model(atom_features, neighbor_features, neighbor_feature_indices,
                                                   crystal_atom_indices) .- target), ps)
        gs = res.grad
        Flux.Optimise.update!(opt, ps, gs)
        training_time += time() - start_time

        @show res.val
    end
    return (training_time=training_time,)
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


trd, vd, ted = FastDEQ.get_materials_project_dataloaders("data/mp/3402/", 512, 512; verbose=true)
model = MaterialsProjectCrystalGraphConvNet(original_atom_feature_length=92, neighbor_feature_length=41) |> gpu
ps = Flux.params(model)
opt = ADAM(0.001)
for e in 1:30
    train_one_epoch!(model, trd, ps, opt, -0.886897307760013f0, 0.9479219363838084f0)
    @show compute_mae(model, vd, -0.886897307760013f0, 0.9479219363838084f0)
    @show compute_mae(model, ted, -0.886897307760013f0, 0.9479219363838084f0)
end

function train_and_validate() end
