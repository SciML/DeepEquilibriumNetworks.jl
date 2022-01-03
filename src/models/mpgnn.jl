struct MaterialsProjectGraphConv{L1,B1,B2}
    atom_feature_length::Int
    neighbor_feature_length::Int
    fc_full::L1
    bn1::B1
    bn2::B2
end

@functor MaterialsProjectGraphConv

function MaterialsProjectGraphConv(atom_feature_length::Int, neighbor_feature_length::Int)
    fc_full = Dense(2 * atom_feature_length + neighbor_feature_length, 2 * atom_feature_length)
    bn1 = BatchNormV2(2 * atom_feature_length)
    bn2 = BatchNormV2(atom_feature_length)
    return MaterialsProjectGraphConv(atom_feature_length, neighbor_feature_length, fc_full, bn1, bn2)
end

function expand_mid(arr::AbstractMatrix, M::Int)
    s1, s2 = size(arr)
    return repeat(reshape(arr, s1, 1, s2), 1, M, 1)
end

function expand_mid(arr::CuMatrix, M::Int)
    s1, s2 = size(arr)
    return reshape(arr, s1, 1, s2) .+ CUDA.zeros(eltype(arr), s1, M, s2)
end

Zygote.@adjoint function expand_mid(arr::CuArray, M::Int)
    s1, s2 = size(arr)
    expand_mid_sensitivity(Δ) = (reshape(sum(Δ; dims=2), s1, s2), nothing)
    return reshape(arr, s1, 1, s2) .+ CUDA.zeros(eltype(arr), s1, M, s2), repeat_no_scalar_sensitivity
end

function (c::MaterialsProjectGraphConv)(atom_in_features::AbstractMatrix{T}, neighbor_features::AbstractArray{T,3},
                                        neighbor_feature_indices::AbstractMatrix{S}) where {T,S<:Int}
    M, N = size(neighbor_feature_indices)
    atom_neighbor_features = atom_in_features[:, neighbor_feature_indices]
    total_neighbor_features = vcat(expand_mid(atom_in_features, M), atom_neighbor_features, neighbor_features)
    total_gated_features = c.fc_full(total_neighbor_features)
    total_gated_features = reshape(c.bn1(reshape(total_gated_features, 2 * c.atom_feature_length, :)),
                                   2 * c.atom_feature_length, M, N)

    neighbor_filter = σ.(total_gated_features[1:(c.atom_feature_length), :, :])
    neighbor_core = softplus.(total_gated_features[(c.atom_feature_length + 1):end, :, :])

    neighbor_sumed = c.bn2(reshape(sum(neighbor_filter .* neighbor_core; dims=2), c.atom_feature_length, N))

    return softplus.(atom_in_features .+ neighbor_sumed), neighbor_features, neighbor_feature_indices
end

struct MaterialsProjectCrystalGraphConvNet{E,C,CF,F}
    embedding::E
    convs::C
    conv_to_fc::CF
    fcs::F
end

@functor MaterialsProjectCrystalGraphConvNet

function MaterialsProjectCrystalGraphConvNet(; original_atom_feature_length::Int, neighbor_feature_length::Int,
                                             atom_feature_length::Int=64, num_conv::Int=3, h_feature_length::Int=128,
                                             n_hidden::Int=1)
    embedding = Dense(original_atom_feature_length, atom_feature_length)
    convs = FChain([MaterialsProjectGraphConv(atom_feature_length, neighbor_feature_length) for _ in 1:num_conv]...)
    conv_to_fc = Dense(atom_feature_length, h_feature_length, softplus)

    fcs = FChain(vcat([Dense(h_feature_length, h_feature_length, softplus) for _ in 1:(n_hidden - 1)],
                      [Dense(h_feature_length, 1)])...)

    return MaterialsProjectCrystalGraphConvNet(embedding, convs, conv_to_fc, fcs)
end

function (c::MaterialsProjectCrystalGraphConvNet)(atom_features::AbstractMatrix{T},
                                                  neighbor_features::AbstractArray{T,3},
                                                  neighbor_feature_indices::AbstractMatrix{S},
                                                  crystal_atom_indices::AbstractVector) where {T,S<:Int}
    atom_features = c.embedding(atom_features)
    atom_features, neighbor_features, neighbor_feature_indices = c.convs(atom_features, neighbor_features,
                                                                         neighbor_feature_indices)
    crystal_features = pool(c, atom_features, crystal_atom_indices)
    crystal_features = c.conv_to_fc(softplus.(crystal_features))
    return c.fcs(crystal_features)
end

function pool(c::MaterialsProjectCrystalGraphConvNet, atom_features::AbstractMatrix{T},
              crystal_atom_indices::AbstractVector) where {T}
    return hcat([mean(atom_features[:, idx_map]; dims=2) for idx_map in crystal_atom_indices]...)
end
