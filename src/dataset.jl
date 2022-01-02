struct MaterialsProjectDataset{D}
    dataset::D
end

struct MaterialsProjectData{AF,NF,NFI,T}
    atom_features::AF
    neighbor_features::NF
    neighbor_feature_indices::NFI
    target::T
end

LearnBase.nobs(d::MaterialsProjectDataset) = length(d.dataset)

LearnBase.getobs(d::MaterialsProjectDataset, i::Int) = d.dataset[i]

DataLoaders.collate(samples::AbstractVector{<:MaterialsProjectData}) = collate_materials_project_dataset(samples)

function get_materials_project_dataloaders(root_dir, train_batchsize, val_batchsize, test_batchsize=val_batchsize;
                                           train_split::AbstractFloat=0.6, val_split::AbstractFloat=0.2, seed::Int=0,
                                           verbose::Bool=false)
    Random.seed!(seed)
    dataset = []
    for f in glob(joinpath(root_dir, "processed_data_*.jls"))
        data = deserialize(f)
        for d in data
            push!(dataset,
                  MaterialsProjectData(permutedims(d[1][1], (2, 1)), permutedims(d[1][2], (3, 2, 1)),
                                       permutedims(d[1][3], (2, 1)) .+ 1, d[2][1]))
        end
        verbose && @info "Loaded processed file $f"
    end

    idxs = randperm(length(dataset))
    idx_1, idx_2 = Int(floor(length(idxs) * train_split)), Int(floor(length(idxs) * val_split))
    train_idxs, val_idxs, test_idxs = idxs[1:idx_1], idxs[(idx_1 + 1):(idx_1 + idx_2)], idxs[(idx_1 + idx_2 + 1):end]
    train_dataset = dataset[train_idxs]
    val_dataset = dataset[val_idxs]
    test_dataset = dataset[test_idxs]

    return DataLoader(MaterialsProjectDataset(train_dataset), train_batchsize; buffered=false),
           DataLoader(MaterialsProjectDataset(val_dataset), val_batchsize; buffered=false),
           DataLoader(MaterialsProjectDataset(test_dataset), test_batchsize; buffered=false)
end

function collate_materials_project_dataset(dataset_list)
    batch_atom_features, batch_neighbor_features, batch_neighbor_indices = [], [], []
    crystal_atom_indices, batch_target = UnitRange{Int64}[], []
    base_idx = 0
    for data in dataset_list
        @unpack atom_features, neighbor_features, neighbor_feature_indices, target = data
        N_i = size(atom_features, 2)
        push!(batch_atom_features, atom_features)
        push!(batch_neighbor_features, neighbor_features)
        push!(batch_neighbor_indices, neighbor_feature_indices .+ base_idx)
        new_indices = (1:N_i) .+ base_idx
        push!(crystal_atom_indices, new_indices)
        push!(batch_target, target)
        base_idx += N_i
    end
    return (f32(hcat(batch_atom_features...)), f32(cat(batch_neighbor_features...; dims=3)),
            Int.(hcat(batch_neighbor_indices...)), crystal_atom_indices, f32(reshape(batch_target, 1, :)))
end
