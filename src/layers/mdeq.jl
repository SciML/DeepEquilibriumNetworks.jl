struct MultiScaleDeepEquilibriumNetwork{N,M1<:Parallel,M2<:MultiParallelNet,RE1,RE2,P,A,K,S} <:
       AbstractDeepEquilibriumNetwork
    main_layers::M1
    mapping_layers::M2
    main_layers_re::RE1
    mapping_layers_re::RE2
    p::P
    ordered_split_idxs::NTuple{N,Int}
    args::A
    kwargs::K
    sensealg::S
    stats::DEQTrainingStats

    function MultiScaleDeepEquilibriumNetwork(main_layers::Parallel, mapping_layers::MultiParallelNet, re1, re2, p,
                                              ordered_split_idxs, args::A, kwargs::K, sensealg::S,
                                              stats) where {A,K,S}
        p_main_layers, re_main_layers = destructure_parameters(main_layers)
        p_mapping_layers, re_mapping_layers = destructure_parameters(mapping_layers)

        ordered_split_idxs = tuple(cumsum([0, length(p_main_layers), length(p_mapping_layers)])...)

        p = p === nothing ? vcat(p_main_layers, p_mapping_layers) : convert(typeof(p_main_layers), p)

        return new{length(ordered_split_idxs),typeof(main_layers),typeof(mapping_layers),typeof(re_main_layers),
                   typeof(re_mapping_layers),typeof(p),A,K,S}(main_layers, mapping_layers, re_main_layers,
                                                              re_mapping_layers, p, ordered_split_idxs, args, kwargs,
                                                              sensealg, stats)
    end
end

Flux.@functor MultiScaleDeepEquilibriumNetwork

function MultiScaleDeepEquilibriumNetwork(main_layers::Tuple, mapping_layers::Matrix, solver; p=nothing,
                                          sensealg=get_default_ssadjoint(0.1f0, 0.1f0, 10), kwargs...)
    @assert size(mapping_layers, 1) == size(mapping_layers, 2) == length(main_layers)

    main_layers = Parallel(flatten_merge, main_layers...)
    mapping_layers = MultiParallelNet(Parallel.(+, map(x -> tuple(x...), eachcol(mapping_layers)))...)

    return MultiScaleDeepEquilibriumNetwork(main_layers, mapping_layers, nothing, nothing, p, nothing, (solver,),
                                            kwargs, sensealg, DEQTrainingStats(0))
end

function (mdeq::MultiScaleDeepEquilibriumNetwork)(x::AbstractArray{T}) where {T}
    z = zero(x)
    initial_conditions = Zygote.@ignore map(l -> l(z), map(l -> l.layers[1], mdeq.mapping_layers.layers))
    u_sizes = Zygote.@ignore size.(initial_conditions)
    u_split_idxs = Zygote.@ignore vcat(0, cumsum(length.(initial_conditions) .÷ size(x, ndims(x)))...)
    u0 = Zygote.@ignore vcat(Flux.flatten.(initial_conditions)...)

    N = length(u_sizes)
    update_is_variational_hidden_dropout_mask_reset_allowed(false)

    function dudt_(u, _p)
        mdeq.stats.nfe += 1

        uₛ = split_array_by_indices(u, u_split_idxs)
        p1, p2 = split_array_by_indices(_p, mdeq.ordered_split_idxs)

        u_reshaped = ntuple(i -> reshape(uₛ[i], u_sizes[i]), N)

        main_layers_output = mdeq.main_layers_re(p1)((u_reshaped[1], x), u_reshaped[2:end]...)

        return mdeq.mapping_layers_re(p2)(main_layers_output)
    end

    dudt(u, _p, t) = vcat(Flux.flatten.(dudt_(u, _p))...) .- u

    ssprob = SteadyStateProblem(dudt, u0, mdeq.p)
    res = solve(ssprob, mdeq.args...; u0=u0, sensealg=mdeq.sensealg, mdeq.kwargs...).u

    x_ = dudt_(res, mdeq.p)
    update_is_variational_hidden_dropout_mask_reset_allowed(true)

    return x_
end
