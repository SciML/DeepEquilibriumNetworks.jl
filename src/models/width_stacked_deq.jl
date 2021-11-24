struct WidthStackedDEQ{has_sdeq,PrB<:BranchNet,D<:Parallel,PoD<:Parallel,CL,F}
    pre_branch_net::PrB
    deqs::D
    post_deq_net::PoD
    combination_layer::CL
    final_mapping::F
end

Flux.@functor WidthStackedDEQ

function WidthStackedDEQ(
    pre_branch_net::BranchNet,
    deqs::Parallel,
    post_deq_net::Parallel,
    combination_layer,
    final_mapping
)
    return WidthStackedDEQ{
        deqs.layers[1] isa SkipDeepEquilibriumNetwork,
        typeof(pre_branch_net),
        typeof(deqs),
        typeof(post_deq_net),
        typeof(combination_layer),
        typeof(final_mapping)
    }(
        pre_branch_net,
        deqs,
        post_deq_net,
        combination_layer,
        final_mapping
    )
end

function WidthStackedDEQ(
    pre_branch_net::Vector,
    deqs::Vector,
    post_deq_net::Vector,
    combination_layer,
    final_mapping
)
    @assert length(pre_branch_net) == length(deqs) == length(post_deq_net)
    _pre_branch_net = BranchNet(pre_branch_net...)
    _deqs = Parallel(flatten_merge, deqs...)
    _post_deq_net = Parallel(flatten_merge, post_deq_net...)
    _combination_layer = combination_layer
    _final_mapping = final_mapping

    return WidthStackedDEQ{
        deqs[1] isa SkipDeepEquilibriumNetwork,
        typeof(_pre_branch_net),
        typeof(_deqs),
        typeof(_post_deq_net),
        typeof(_combination_layer),
        typeof(_final_mapping)
    }(
        _pre_branch_net,
        _deqs,
        _post_deq_net,
        _combination_layer,
        _final_mapping
    )
end

function WidthStackedDEQ(
    pre_branch_net::NTuple{N,PrB},
    deqs::NTuple{N,D},
    post_deq_net::NTuple{N,PoD},
    combination_layer::CL,
    final_mapping::F
) where {N,PrB,D,PoD,CL,F}
    _pre_branch_net = BranchNet(pre_branch_net...)
    _deqs = Parallel(flatten_merge, deqs...)
    _post_deq_net = Parallel(flatten_merge, post_deq_net...)
    _combination_layer = combination_layer
    _final_mapping = final_mapping

    return WidthStackedDEQ{
        deqs[1] isa SkipDeepEquilibriumNetwork,
        typeof(_pre_branch_net),
        typeof(_deqs),
        typeof(_post_deq_net),
        typeof(_combination_layer),
        typeof(_final_mapping)
    }(
        _pre_branch_net,
        _deqs,
        _post_deq_net,
        _combination_layer,
        _final_mapping
    )
end

function Flux.gpu(ws::WidthStackedDEQ{S}) where {S}
    prbn = Flux.gpu(ws.pre_branch_net)
    deq = Flux.gpu(ws.deqs)
    pobn = Flux.gpu(ws.post_deq_net)
    cl = Flux.gpu(ws.combination_layer)
    fm = Flux.gpu(ws.final_mapping)
    return WidthStackedDEQ{S,typeof(prbn),typeof(deq),typeof(pobn),typeof(cl),typeof(fm)}(
        prbn,
        deq,
        pobn,
        cl,
        fm
    )
end

function Flux.cpu(ws::WidthStackedDEQ{S}) where {S}
    prbn = Flux.cpu(ws.pre_branch_net)
    deq = Flux.cpu(ws.deqs)
    pobn = Flux.cpu(ws.post_deq_net)
    cl = Flux.cpu(ws.combination_layer)
    fm = Flux.cpu(ws.final_mapping)
    return WidthStackedDEQ{S,typeof(prbn),typeof(deq),typeof(pobn),typeof(cl),typeof(fm)}(
        prbn,
        deq,
        pobn,
        cl,
        fm
    )
end

function (wsdeq::WidthStackedDEQ{false})(x::AbstractArray)
    deq_inputs = wsdeq.pre_branch_net(x)
    deq_outputs = wsdeq.deqs(deq_inputs...)
    post_deq_outputs = wsdeq.post_deq_net(deq_outputs...)
    return wsdeq.final_mapping(wsdeq.combination_layer(post_deq_outputs...))
end

function (wsdeq::WidthStackedDEQ{true})(x::AbstractArray)
    deq_inputs = wsdeq.pre_branch_net(x)
    deq_outputs = wsdeq.deqs(deq_inputs...)
    post_deq_outputs = wsdeq.post_deq_net(first.(deq_outputs)...)
    return wsdeq.final_mapping(wsdeq.combination_layer(post_deq_outputs...)), deq_outputs
end

FastDEQ.get_and_clear_nfe!(model::WidthStackedDEQ) =
    get_and_clear_nfe!.(model.deqs.layers)
