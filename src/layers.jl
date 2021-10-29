Flux.trainable(deq::AbstractDeepEquilibriumNetwork) = (deq.p,)

function get_and_clear_nfe!(model::AbstractDeepEquilibriumNetwork)
    nfe = model.stats.nfe
    model.stats.nfe = 0
    return nfe
end

mutable struct DEQTrainingStats
    nfe::Int
end

struct DeepEquilibriumNetwork{M,P,RE,A,S,K} <: AbstractDeepEquilibriumNetwork
    model::M
    p::P
    re::RE
    args::A
    kwargs::K
    sensealg::S
    stats::DEQTrainingStats
end

Flux.@functor DeepEquilibriumNetwork

function DeepEquilibriumNetwork(
    model,
    args...;
    p = nothing,
    sensealg = SteadyStateAdjoint(
        autodiff = false,
        autojacvec = ZygoteVJP(),
        linsolve = LinSolveKrylovJL(rtol = 0.1f0, atol = 0.1f0),
    ),
    kwargs...,
)
    _p, re = Flux.destructure(model)
    p = p === nothing ? _p : p
    return DeepEquilibriumNetwork(
        model,
        p,
        re,
        args,
        kwargs,
        sensealg,
        DEQTrainingStats(0),
    )
end

function (deq::DeepEquilibriumNetwork)(x::AbstractArray{T}, p = deq.p) where {T}
    # Solving the equation f(u) - u = du = 0
    z = zero(x)

    function dudt(u, _p, t)
        deq.stats.nfe += 1
        return deq.re(_p)(u, x) .- u
    end

    ssprob = SteadyStateProblem(dudt, z, p)
    sol = solve(
        ssprob,
        deq.args...;
        u0 = z,
        sensealg = deq.sensealg,
        deq.kwargs...,
    )
    deq.stats.nfe += 1
    return deq.re(p)(sol.u, x)::typeof(x)
end

struct SkipDeepEquilibriumNetwork{M,S,P,RE1,RE2,A,Se,K} <:
       AbstractDeepEquilibriumNetwork
    model::M
    shortcut::S
    p::P
    re1::RE1
    re2::RE2
    split_idx::Int
    args::A
    kwargs::K
    sensealg::Se
    stats::DEQTrainingStats
end

Flux.@functor SkipDeepEquilibriumNetwork

function SkipDeepEquilibriumNetwork(
    model,
    shortcut,
    args...;
    p = nothing,
    sensealg = SteadyStateAdjoint(
        autodiff = false,
        autojacvec = ZygoteVJP(),
        linsolve = LinSolveKrylovJL(rtol = 0.1f0, atol = 0.1f0),
    ),
    kwargs...,
)
    p1, re1 = Flux.destructure(model)
    p2, re2 = Flux.destructure(shortcut)
    p = p === nothing ? vcat(p1, p2) : p
    return SkipDeepEquilibriumNetwork(
        model,
        shortcut,
        p,
        re1,
        re2,
        length(p1),
        args,
        kwargs,
        sensealg,
        DEQTrainingStats(0),
    )
end

function (deq::SkipDeepEquilibriumNetwork)(
    x::AbstractArray{T},
    p = deq.p,
) where {T}
    p1, p2 = p[1:deq.split_idx], p[deq.split_idx+1:end]
    z = deq.re2(p2)(x)::typeof(x)
    deq.stats.nfe += 1

    # Solving the equation f(u) - u = du = 0
    function dudt(u, _p, t)
        deq.stats.nfe += 1
        return deq.re1(_p)(u, x) .- u
    end

    ssprob = SteadyStateProblem(dudt, z, p1)
    u = solve(
        ssprob,
        deq.args...;
        u0 = z,
        sensealg = deq.sensealg,
        deq.kwargs...,
    ).u::typeof(x)
    res = deq.re1(p1)(u, x)::typeof(x)
    deq.stats.nfe += 1
    return res, z
end


## FIXME: Zygote being dumb keeps complaining about mutating arrays for general implementation
##        of MDEQs. Since we will use depth 4 MDEQs, I will only implement those.
struct MultiScaleDeepEquilibriumNetwork{M1,M2,RE1,RE2,P,A,K,S} <:
       AbstractDeepEquilibriumNetwork
    main_layers::M1
    mapping_layers::M2
    main_layers_re::RE1
    mapping_layers_re::RE2
    p::P
    ordered_split_idxs::Vector{Int}
    args::A
    kwargs::K
    sensealg::S
    stats::DEQTrainingStats
end

Flux.@functor MultiScaleDeepEquilibriumNetwork

function MultiScaleDeepEquilibriumNetwork(
    main_layers::Tuple,
    mapping_layers::Tuple,
    solver;
    p = nothing,
    sensealg = SteadyStateAdjoint(
        autodiff = false,
        autojacvec = ZygoteVJP(),
        linsolve = LinSolveKrylovJL(rtol = 0.1f0, atol = 0.1f0),
    ),
    kwargs...,
)
    main_layers_res = []
    mapping_layers_res = []
    ordered_split_idxs = [0]
    c = 0
    ps = []
    for layer in main_layers
        _p, _re = Flux.destructure(layer)
        push!(main_layers_res, _re)
        push!(ps, _p)
        c += length(_p)
        push!(ordered_split_idxs, c)
    end
    for layers in mapping_layers
        layer_list = []
        for layer in layers
            _p, _re = Flux.destructure(layer)
            push!(layer_list, _re)
            push!(ps, _p)
            c += length(_p)
            push!(ordered_split_idxs, c)
        end
        push!(mapping_layers_res, tuple(layer_list...))
    end
    p = p === nothing ? vcat(ps...) : p
    return MultiScaleDeepEquilibriumNetwork(
        main_layers,
        mapping_layers,
        tuple(main_layers_res...),
        tuple(mapping_layers_res...),
        p,
        ordered_split_idxs,
        (solver,),
        kwargs,
        sensealg,
        DEQTrainingStats(0),
    )
end

function (mdeq::MultiScaleDeepEquilibriumNetwork)(
    x::AbstractArray{T,N},
    p = mdeq.p,
) where {T,N}
    initial_conditions =
        Zygote.@ignore Vector{SingleResolutionFeatures{typeof(vec(x)),T}}(
            undef,
            length(mdeq.main_layers),
        )
    sizes =
        Zygote.@ignore Vector{NTuple{N,Int64}}(undef, length(mdeq.main_layers))
    _z = zero(x)
    Zygote.@ignore for i = 1:length(initial_conditions)
        _x = mdeq.mapping_layers[1][i](_z)
        sizes[i] = size(_x)
        initial_conditions[i] = SingleResolutionFeatures(vec(_x))
    end
    u0 = Zygote.@ignore construct(MultiResolutionFeatures, initial_conditions)

    function dudt(u, _p, t)
        mdeq.stats.nfe += 1

        u_prevs =
            [reshape(u.nodes[i].values, sizes[i]) for i = 1:length(u.nodes)]

        counter = 1

        function apply_main_layer(i)
            layer = mdeq.main_layers_re[i](
                _p[mdeq.ordered_split_idxs[i]+1:mdeq.ordered_split_idxs[i+1]],
            )
            counter += 1
            return i == 1 ? layer(u_prevs[i], x) : layer(u_prevs[i])
        end

        buffer = map(apply_main_layer, 1:length(mdeq.main_layers))

        function apply_mapping_layer(i)
            layer = mdeq.mapping_layers_re[i](
                _p[mdeq.ordered_split_idxs[i]+1:mdeq.ordered_split_idxs[i+1]],
            )
            counter += 1
            return i == 1 ? layer(buffer, x) : layer(buffer)
        end

        idxs = permutedims(
            CartesianIndices((1:length(mdeq.main_layers), 1:length(mdeq.main_layers))),
            (2, 1)
        )

        function apply_mapping_layer(idx::CartesianIndex)
            i, j = idx.I
            layer = mdeq.mapping_layers_re[i][j](
                _p[mdeq.ordered_split_idxs[counter]+1:mdeq.ordered_split_idxs[counter+1]],
            )
            counter += 1
            return layer(buffer[i])
        end

        res = map(apply_mapping_layer, idxs)

        accum = sum.(eachrow(res))

        return construct(
            MultiResolutionFeatures,
            SingleResolutionFeatures.(vec.(accum)),
        )
    end

    ssprob = SteadyStateProblem(dudt, u0, p)
    sol =
        dudt(
            solve(
                ssprob,
                mdeq.args...;
                u0 = u0,
                sensealg = mdeq.sensealg,
                mdeq.kwargs...,
            ).u,
            p,
            0,
        ).nodes

    return [reshape(sol[i].values, sizes[i]) for i = 1:length(sol)]
end
