modeltype_to_val(::DeepEquilibriumNetwork) = Val(1)
modeltype_to_val(::SkipDeepEquilibriumNetwork) = Val(2)
modeltype_to_val(::MultiScaleDeepEquilibriumNetworkS4) = Val(3)
modeltype_to_val(::MultiScaleSkipDeepEquilibriumNetworkS4) = Val(4)
modeltype_to_val(m) = Val(-1)

# Default to nothing happening
reset_mask!(x) = nothing

struct DEQChain{V,P1,D,P2}
    pre_deq::P1
    deq::D
    post_deq::P2

    function DEQChain(layers...)
        if length(layers) == 3
            pre_deq, deq, post_deq = layers
            val = modeltype_to_val(deq)
            val == Val(-1) && error(
                "$deq must subtype AbstractDeepEquilibriumNetwork and define `modeltype_to_val`",
            )
            return new{val,typeof(pre_deq),typeof(deq),typeof(post_deq)}(
                pre_deq,
                deq,
                post_deq,
            )
        end
        pre_deq = []
        post_deq = []
        deq = nothing
        encounter_deq = false
        global_val = Val(1)
        for l in layers
            val = modeltype_to_val(l)
            if val != Val(-1)
                global_val = val
                encounter_deq &&
                    error("Can have only 1 DEQ Layer in the Chain!!!")
                deq = l
                encounter_deq = true
                continue
            end
            if encounter_deq
                push!(post_deq, l)
            else
                push!(pre_deq, l)
            end
        end
        !encounter_deq &&
            error("No DEQ Layer in the Chain!!! Maybe you wanted to use Chain")
        pre_deq = length(pre_deq) == 0 ? identity : Chain(pre_deq...)
        post_deq = length(post_deq) == 0 ? identity : Chain(post_deq...)
        return new{global_val,typeof(pre_deq),typeof(deq),typeof(post_deq)}(
            pre_deq,
            deq,
            post_deq,
        )
    end
end

Flux.@functor DEQChain

(deq::Union{DEQChain{Val(1)},DEQChain{Val(3)}})(x) =
    deq.post_deq(deq.deq(deq.pre_deq(x)))

function (deq::Union{DEQChain{Val(2)},DEQChain{Val(4)}})(x)
    x1 = deq.pre_deq(x)
    z, ẑ = deq.deq(x1)
    x2 = deq.post_deq(z)
    return (x2, (z, ẑ))
end

function get_and_clear_nfe!(model::DEQChain)
    nfe = model.deq.stats.nfe
    model.deq.stats.nfe = 0
    return nfe
end


# Clean Chain
struct Sequential{C<:Chain}
    flattened_chain::C
    function Sequential(chain::Chain)
        _chain = _recursively_flatten(chain)
        return new{typeof(_chain)}(_chain)
    end
    function Sequential(layers...)
        return Sequential(Chain(layers...))
    end
end

Flux.@functor Sequential

_recursively_flatten(x; kwargs...) = x

function _recursively_flatten(c::Chain; depth::Int = 0)
    if depth > 0
        return vcat(_recursively_flatten.(c.layers; depth = depth + 1)...)
    else
        return Chain(
            vcat(_recursively_flatten.(c.layers; depth = depth + 1)...)...,
        )
    end
end

(s::Sequential)(x) = s.flattened_chain(x)

Flux.gpu(s::Sequential) = Sequential(Chain(gpu.(s.flattened_chain.layers)...))


# Crystal Graph CNN
struct CrystalGraphCNN{deq_type,PD1,D,PD2,P,DC}
    pre_deq::PD1
    deq::D
    post_deq::PD2
    pool::P
    dense_chain::DC
end

Flux.@functor CrystalGraphCNN

function Flux.gpu(c::CrystalGraphCNN{DT}) where {DT}
    pd1 = gpu(c.pre_deq)
    deq = gpu(c.deq)
    pd2 = gpu(c.post_deq)
    p = gpu(c.pool)
    d = gpu(c.dense_chain)
    return CrystalGraphCNN{
        DT,
        typeof(pd1),
        typeof(deq),
        typeof(pd2),
        typeof(p),
        typeof(d),
    }(
        pd1,
        deq,
        pd2,
        p,
        d,
    )
end


function CrystalGraphCNN(
    input_feature_length::Int;
    num_conv::Int = 2,
    conv_activation = softplus,
    atom_conv_feature_length::Int = 80,
    pool_type::Symbol = :mean,
    pool_width::Real = 0.1,
    pooled_feature_length::Int = 40,
    num_hidden_layers::Int = 1,
    hidden_layer_activation = softplus,
    output_layer_activation = identity,
    output_length::Int = 1,
    initW = Flux.glorot_uniform,
    deq_type::Union{Nothing,Symbol} = nothing,
)
    @assert atom_conv_feature_length >= pooled_feature_length "Feature length after pooling must be <= feature length before pooling!"
    @assert deq_type ∈ (nothing, :deq, :skip_deq)

    pre_deq = AGNConv(
        input_feature_length => atom_conv_feature_length,
        conv_activation,
        initW = initW,
    )

    if deq_type === nothing
        deq = identity
        post_deq = tuple(
            [
                AGNConv(
                    atom_conv_feature_length => atom_conv_feature_length,
                    conv_activation,
                    initW = initW,
                ) for i = 1:num_conv-1
            ]...,
        )
    elseif deq_type == :deq
        error("Not yet implemented")
    elseif deq_type == :skip_deq
        error("Not yet implemented")
    end

    pool = AGNPool(
        pool_type,
        atom_conv_feature_length,
        pooled_feature_length,
        pool_width,
    )

    dense_chain = Sequential(
        Chain(
            [
                Dense(
                    pooled_feature_length,
                    pooled_feature_length,
                    hidden_layer_activation,
                    init = initW,
                ) for i = 1:num_hidden_layers-1
            ]...,
            Dense(
                pooled_feature_length,
                output_length,
                output_layer_activation,
                init = initW,
            ),
        ),
    )

    return CrystalGraphCNN{
        modeltype_to_val(deq),
        typeof(pre_deq),
        typeof(deq),
        typeof(post_deq),
        typeof(pool),
        typeof(dense_chain),
    }(
        pre_deq,
        deq,
        post_deq,
        pool,
        dense_chain,
    )
end

function (cgcnn::CrystalGraphCNN)(inputs::Tuple)
    lapl, x = inputs
    _lapl1, _x1 = cgcnn.pre_deq(lapl, x)
    _lapl2, _x2 = cgcnn.deq((_lapl1, _x1))
    _lapl3, _x3 = cgcnn.post_deq[1](_lapl2, _x2)
    for layer in cgcnn.post_deq[2:end]
        _lapl3, _x3 = layer(_lapl3, _x3)
    end
    _x4 = cgcnn.pool(_lapl3, _x3)
    return cgcnn.dense_chain(_x4)
end

function (cgcnn::CrystalGraphCNN)(bag::BatchedAtomicGraph)
    lapl, x = bag.laplacians, bag.encoded_features
    _lapl1, _x1 = cgcnn.pre_deq(lapl, x)
    _lapl2, _x2 = cgcnn.deq((_lapl1, _x1))
    _lapl3, _x3 = cgcnn.post_deq[1](_lapl2, _x2)
    for layer in cgcnn.post_deq[2:end]
        _lapl3, _x3 = layer(_lapl3, _x3)
    end
    _x4 = hcat([cgcnn.pool(_lapl3[s1+1:s2,s1+1:s2], _x3[:,s1+1:s2]) for (s1, s2) in zip(bag.sizes[1:end-1],bag.sizes[2:end])]...)
    return cgcnn.dense_chain(_x4)
end