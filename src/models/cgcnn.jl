using FluxExperimental: AGNConv, AGNPool

(gn::GroupNormV2)(lapl::AbstractMatrix, X::AbstractMatrix) = (lapl, gn(X))
(bn::BatchNormV2)(lapl::AbstractMatrix, X::AbstractMatrix) = (lapl, bn(X))
# Update FluxExperimental and remove this
(bn::BatchNormV2)(lapl::CuMatrix{T}, X::AbstractMatrix) where {T<:Union{Float32,Float64}} = (lapl, bn(X))

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

function Base.show(io::IO, model::CrystalGraphCNN)
    l1 = length(destructure_parameters(model)[1])
    println(io, "CrystalGraphCNN(")
    print(io, "\t")
    show(io, model.pre_deq)
    print(io, "\n\t")
    show(io, model.deq)
    print(io, "\n\t")
    show(io, model.post_deq)
    print(io, "\n\t")
    show(io, model.pool)
    print(io, "\n\t")
    show(io, model.dense_chain)
    return print(io, "\n) $l1 Trainable Parameters")
end

function CrystalGraphCNN(input_feature_length::Int; num_conv::Int=2, conv_activation=gelu,
                         atom_conv_feature_length::Int=80, pool_type::Symbol=:mean, pool_width::Real=0.1,
                         pooled_feature_length::Int=40, num_hidden_layers::Int=1, hidden_layer_activation=gelu,
                         output_layer_activation=identity, output_length::Int=1, initW_explicit=Flux.glorot_uniform,
                         initW_deq=normal_init(), deq_type::Symbol=:explicit, abstol::Real=0.1f0, reltol::Real=0.1f0,
                         ode_solver=BS3(), maxiters::Int=10, solver=get_default_dynamicss_solver(reltol, abstol, ode_solver))
    @assert atom_conv_feature_length >= pooled_feature_length "Feature length after pooling must be <= feature length before pooling!"
    @assert deq_type ∈ (:explicit, :deq, :skip_deq, :skip_deq_no_extra_params) "Unknown deq_type: $(deq_type)!"

    pre_deq = FChain(AGNConv(input_feature_length => atom_conv_feature_length; initW=initW_explicit),
                    BatchNormV2(atom_conv_feature_length, conv_activation; track_stats=true, affine=true),
                    AGNConv(atom_conv_feature_length => atom_conv_feature_length; initW=initW_explicit),
                    BatchNormV2(atom_conv_feature_length, conv_activation; track_stats=true, affine=true))

    if deq_type == :explicit
        deq = NoOpLayer()
        layers = []
        for i in 1:(num_conv - 2)
            push!(layers,
                  AGNConv(atom_conv_feature_length => atom_conv_feature_length; initW=initW_explicit,
                          initb=(args...) -> Flux.Zeros()))
            push!(layers, GroupNormV2(atom_conv_feature_length, 8, conv_activation; track_stats=false, affine=true))
        end
        push!(layers,
              AGNConv(atom_conv_feature_length => atom_conv_feature_length, conv_activation; initW=initW_explicit,
                      initb=(args...) -> Flux.Zeros()))
        post_deq = FChain(layers...)
    elseif deq_type == :deq
        deq = DeepEquilibriumNetwork(ResidualAGNConvBlock(+,
                                                          AGNConv(atom_conv_feature_length => atom_conv_feature_length,
                                                                  conv_activation; initW=initW_deq),
                                                          AGNConv(atom_conv_feature_length => atom_conv_feature_length,
                                                                  conv_activation; initW=initW_deq)),
                                     solver;
                                     sensealg=get_default_ssadjoint(reltol, abstol, maxiters), verbose=false,
                                     maxiters=maxiters)
        post_deq = NoOpLayer()
    elseif deq_type == :skip_deq
        deq = SkipDeepEquilibriumNetwork(ResidualAGNConvBlock(+,
                                                              AGNConv(atom_conv_feature_length => atom_conv_feature_length,
                                                                      conv_activation; initW=initW_deq),
                                                              AGNConv(atom_conv_feature_length => atom_conv_feature_length,
                                                                      conv_activation; initW=initW_deq)),
                                         AGNConv(atom_conv_feature_length => atom_conv_feature_length, conv_activation;
                                                 initW=initW), solver;
                                         sensealg=get_default_ssadjoint(reltol, abstol, maxiters), verbose=false,
                                         maxiters=maxiters)
        post_deq = NoOpLayer()
    elseif deq_type == :skip_deq_no_extra_params
        deq = SkipDeepEquilibriumNetwork(ResidualAGNConvBlock(+,
                                                              AGNConv(atom_conv_feature_length => atom_conv_feature_length,
                                                                      conv_activation; initW=initW_deq),
                                                              AGNConv(atom_conv_feature_length => atom_conv_feature_length,
                                                                      conv_activation; initW=initW_deq)),
                                         solver;
                                         sensealg=get_default_ssadjoint(reltol, abstol, maxiters), verbose=false,
                                         maxiters=maxiters)
        post_deq = NoOpLayer()
    end

    pool = AGNPool(pool_type, atom_conv_feature_length => pooled_feature_length, pool_width)

    layers = []
    for i in 1:(num_hidden_layers - 1)
        push!(layers, Dense(pooled_feature_length, pooled_feature_length; init=initW_explicit))
        push!(layers, BatchNormV2(pooled_feature_length, hidden_layer_activation; track_stats=true, affine=true))
    end

    dense_chain = FChain(layers...,
                        Dense(pooled_feature_length, output_length, output_layer_activation; init=initW_explicit))

    return CrystalGraphCNN(pre_deq, deq, post_deq, pool, dense_chain)
end

function CrystalGraphCNN(pre_deq, deq, post_deq, pool, dense_chain)
    return CrystalGraphCNN{modeltype_to_val(deq),typeof(pre_deq),typeof(deq),typeof(post_deq),typeof(pool),
                           typeof(dense_chain)}(pre_deq, deq, post_deq, pool, dense_chain)
end

function (cgcnn::CrystalGraphCNN)(bag::BatchedAtomicGraph)
    lapl, x = bag.laplacians, bag.encoded_features
    # debug_backward_pass("Pre DEQ")
    _lapl1, _x1 = cgcnn.pre_deq(lapl, x)
    # debug_backward_pass("DEQ")
    _lapl2, _x2 = cgcnn.deq(_lapl1, _x1)
    # debug_backward_pass("Post DEQ")
    _lapl3, _x3 = cgcnn.post_deq(_lapl2, _x2)
    # debug_backward_pass("Pooling")
    _x4 = hcat([cgcnn.pool(_lapl3[(s1 + 1):s2, (s1 + 1):s2], _x3[:, (s1 + 1):s2])[2]
                for (s1, s2) in zip(bag.sizes[1:(end - 1)], bag.sizes[2:end])]...)
    # debug_backward_pass("Final Mapping")
    r = cgcnn.dense_chain(_x4)
    # debug_backward_pass("Done")
    return r
end

function (cgcnn::CrystalGraphCNN{Val(2)})(bag::BatchedAtomicGraph)
    lapl, x = bag.laplacians, bag.encoded_features
    _lapl1, _x1 = cgcnn.pre_deq(lapl, x)
    (_lapl2, _x2), guess = cgcnn.deq((_lapl1, _x1))
    _lapl3, _x3 = cgcnn.post_deq(_lapl2, _x2)
    _x4 = hcat([cgcnn.pool(_lapl3[(s1 + 1):s2, (s1 + 1):s2], _x3[:, (s1 + 1):s2])[2]
                for (s1, s2) in zip(bag.sizes[1:(end - 1)], bag.sizes[2:end])]...)
    return cgcnn.dense_chain(_x4), (_x2, guess)
end

function get_and_clear_nfe!(model::CrystalGraphCNN)
    nfe = model.deq.stats.nfe
    model.deq.stats.nfe = 0
    return nfe
end

get_and_clear_nfe!(model::CrystalGraphCNN{Val(-1)}) = -1

function (lc::SupervisedLossContainer)(model::CrystalGraphCNN{Val(2)}, x, y; kwargs...)
    ŷ, guess_pair = model(x; kwargs...)
    return lc.loss_function(ŷ, y) + lc.λ * mean(abs, guess_pair[1] .- guess_pair[2])
end
