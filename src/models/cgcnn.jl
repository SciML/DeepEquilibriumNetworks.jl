# Residual AGN Modules
struct ResidualAGNConvBlock{Op,C1<:AGNConv,C2<:AGNConv}
    op::Op
    c1::C1
    c2::C2
end

Flux.@functor ResidualAGNConvBlock

function (rconv::ResidualAGNConvBlock)(lapl, x1, x2)
    cx1 = rconv.c1(lapl, x1)[2]
    cx2 = rconv.c2(lapl, x2)[2]
    return lapl, rconv.op(cx1, cx2)
end

# Crystal Graph CNN
struct CrystalGraphCNN{deq_type,PD1,D,PD2,P,DC}
    pre_deq::PD1
    deq::D
    post_deq::PD2
    pool::P
    dense_chain::DC
end

Flux.@functor CrystalGraphCNN

function CrystalGraphCNN(input_feature_length::Int; num_conv::Int=2, conv_activation=softplus,
                         atom_conv_feature_length::Int=80, pool_type::Symbol=:mean, pool_width::Real=0.1,
                         pooled_feature_length::Int=40, num_hidden_layers::Int=1, hidden_layer_activation=softplus,
                         output_layer_activation=identity, output_length::Int=1, initW=Flux.glorot_uniform,
                         deq_type::Symbol=:explicit, abstol::Real=0.1f0, reltol::Real=0.1f0, ode_solver=Tsit5(),
                         maxiters::Int=10)
    @assert atom_conv_feature_length >= pooled_feature_length "Feature length after pooling must be <= feature length before pooling!"
    @assert deq_type ∈ (:explicit, :deq, :skip_deq, :skip_deq_no_extra_params) "Unknown deq_type: $(deq_type)!"

    pre_deq = AGNConv(input_feature_length => atom_conv_feature_length, conv_activation; initW=initW)

    if deq_type == :explicit
        deq = identity
        post_deq = tuple([AGNConv(atom_conv_feature_length => atom_conv_feature_length, conv_activation; initW=initW)
                          for i in 1:(num_conv - 1)]...)
    elseif deq_type == :deq
        deq = DeepEquilibriumNetwork(ResidualAGNConvBlock(+,
                                                          AGNConv(atom_conv_feature_length => atom_conv_feature_length,
                                                                  conv_activation; initW=initW),
                                                          AGNConv(atom_conv_feature_length => atom_conv_feature_length,
                                                                  conv_activation; initW=initW)),
                                     get_default_dynamicss_solver(reltol, abstol, ode_solver);
                                     sensealg=get_default_ssadjoint(reltol, abstol, maxiters), verbose=false,
                                     maxiters=maxiters)
        post_deq = ((x, y) -> (x, y),)
    elseif deq_type == :skip_deq
        deq = SkipDeepEquilibriumNetwork(ResidualAGNConvBlock(+,
                                                              AGNConv(atom_conv_feature_length => atom_conv_feature_length,
                                                                      conv_activation; initW=initW),
                                                              AGNConv(atom_conv_feature_length => atom_conv_feature_length,
                                                                      conv_activation; initW=initW)),
                                         AGNConv(atom_conv_feature_length => atom_conv_feature_length, conv_activation;
                                                 initW=initW), get_default_dynamicss_solver(reltol, abstol, ode_solver);
                                         sensealg=get_default_ssadjoint(reltol, abstol, maxiters), verbose=false,
                                         maxiters=maxiters)
        post_deq = ((x, y) -> (x, y),)
    elseif deq_type == :skip_deq_no_extra_params
        deq = SkipDeepEquilibriumNetwork(ResidualAGNConvBlock(+,
                                                              AGNConv(atom_conv_feature_length => atom_conv_feature_length,
                                                                      conv_activation; initW=initW),
                                                              AGNConv(atom_conv_feature_length => atom_conv_feature_length,
                                                                      conv_activation; initW=initW)),
                                         get_default_dynamicss_solver(reltol, abstol, ode_solver);
                                         sensealg=get_default_ssadjoint(reltol, abstol, maxiters), verbose=false,
                                         maxiters=maxiters)
        post_deq = ((x, y) -> (x, y),)
    end

    pool = AGNPool(pool_type, atom_conv_feature_length, pooled_feature_length, pool_width)

    dense_chain = Chain([Dense(pooled_feature_length, pooled_feature_length, hidden_layer_activation; init=initW)
                         for i in 1:(num_hidden_layers - 1)]...,
                        Dense(pooled_feature_length, output_length, output_layer_activation; init=initW))

    return CrystalGraphCNN(pre_deq, deq, post_deq, pool, dense_chain)
end

function CrystalGraphCNN(pre_deq, deq, post_deq, pool, dense_chain)
    return CrystalGraphCNN{modeltype_to_val(deq),typeof(pre_deq),typeof(deq),typeof(post_deq),typeof(pool),
                           typeof(dense_chain)}(pre_deq, deq, post_deq, pool, dense_chain)
end

function (cgcnn::CrystalGraphCNN)(bag::BatchedAtomicGraph)
    lapl, x = bag.laplacians, bag.encoded_features
    _lapl1, _x1 = cgcnn.pre_deq(lapl, x)
    _lapl2, _x2 = cgcnn.deq((_lapl1, _x1))
    _lapl3, _x3 = cgcnn.post_deq[1](_lapl2, _x2)
    for layer in cgcnn.post_deq[2:end]
        _lapl3, _x3 = layer(_lapl3, _x3)
    end
    _x4 = hcat([cgcnn.pool(_lapl3[(s1 + 1):s2, (s1 + 1):s2], _x3[:, (s1 + 1):s2])
                for (s1, s2) in zip(bag.sizes[1:(end - 1)], bag.sizes[2:end])]...)
    return cgcnn.dense_chain(_x4)
end

function (cgcnn::CrystalGraphCNN{Val(2)})(bag::BatchedAtomicGraph)
    lapl, x = bag.laplacians, bag.encoded_features
    _lapl1, _x1 = cgcnn.pre_deq(lapl, x)
    (_lapl2, _x2), guess = cgcnn.deq((_lapl1, _x1))
    _lapl3, _x3 = cgcnn.post_deq[1](_lapl2, _x2)
    for layer in cgcnn.post_deq[2:end]
        _lapl3, _x3 = layer(_lapl3, _x3)
    end
    _x4 = hcat([cgcnn.pool(_lapl3[(s1 + 1):s2, (s1 + 1):s2], _x3[:, (s1 + 1):s2])
                for (s1, s2) in zip(bag.sizes[1:(end - 1)], bag.sizes[2:end])]...)
    return cgcnn.dense_chain(_x4), (_x2, guess)
end

function get_and_clear_nfe!(model::CrystalGraphCNN)
    nfe = model.deq.stats.nfe
    model.deq.stats.nfe = 0
    return nfe
end

get_and_clear_nfe!(model::CrystalGraphCNN{Val(-1)}) = -1

