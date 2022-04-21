struct MultiScaleDeepEquilibriumNetwork{N,L,M,A,S,K} <: AbstractDeepEquilibriumNetwork
    model::M
    solver::A
    sensealg::S
    scales::NTuple{N,NTuple{L,Int64}}
    kwargs::K
end

function initialstates(rng::AbstractRNG, deq::MultiScaleDeepEquilibriumNetwork)
    return (model=initialstates(rng, deq.model), split_idxs=Tuple(vcat(0, cumsum(prod.(deq.scales))...)), fixed_depth=0)
end

function MultiScaleDeepEquilibriumNetwork(
    main_layers::Tuple,
    mapping_layers::Matrix,
    post_fuse_layer::Union{Nothing,Tuple},
    solver,
    scales;
    sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10),
    kwargs...,
)
    l1 = ExplicitFluxLayers.Parallel(nothing, main_layers...)
    l2 = ExplicitFluxLayers.BranchLayer(
        ExplicitFluxLayers.Parallel.(+, map(x -> tuple(x...), eachrow(mapping_layers))...)...
    )
    model = if post_fuse_layer === nothing
        ExplicitFluxLayers.Chain(l1, l2)
    else
        l3 = ExplicitFluxLayers.Parallel(nothing, post_fuse_layer...)
        ExplicitFluxLayers.Chain(l1, l2, l3)
    end
    return MultiScaleDeepEquilibriumNetwork(model, solver, sensealg, scales, kwargs)
end

function get_initial_condition_mdeq(scales::NTuple, x::AbstractArray{T,N}, st::NamedTuple{fields}) where {T,N,fields}
    if hasproperty(st, :initial_condition) && size(st.initial_condition, 2) == size(x, N)
        return st.initial_condition, st
    end
    u0 = vcat(map(scale -> fill!(similar(x, prod(scale), size(x, N)), T(0)), scales)...)
    st = merge((initial_condition=u0,), st)
    return u0, st
end

Zygote.@nograd get_initial_condition_mdeq

function (deq::MultiScaleDeepEquilibriumNetwork{N})(
    x::AbstractArray{T}, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
) where {N,T}
    z, st = get_initial_condition_mdeq(deq.scales, x, st)

    if !iszero(st.fixed_depth)
        z_star = split_and_reshape(z, st.split_idxs, deq.scales)
        st_ = st.model

        for _ in 1:(st.fixed_depth)
            z_star, st_ = deq.model(((z_star[1], x), z_star[2:N]...), ps, st_)
        end

        @set! st.model = st_

        return (z_star, DeepEquilibriumSolution(vcat(Flux.flatten.(z_star)...), z, z, 0.0f0, st.fixed_depth)), st
    end

    function dudt_(u, p, t)
        u_split = split_and_reshape(u, st.split_idxs, deq.scales)
        u_, st_ = deq.model(((u_split[1], x), u_split[2:N]...), p, st.model)
        return u_, st_
    end

    dudt(u, p, t) = vcat(Flux.flatten.(dudt_(u, p, t)[1])...) .- u

    prob = SteadyStateProblem(ODEFunction{false}(dudt), z, ps)
    sol = solve(prob, deq.solver; sensealg=deq.sensealg, deq.kwargs...)
    z_star, st_ = dudt_(sol.u, ps, nothing)

    residual = dudt(sol.u, ps, nothing)

    @set! st.model = st_

    return (
        (z_star, DeepEquilibriumSolution(vcat(Flux.flatten.(z_star)...), z, residual, 0.0f0, sol.destats.nf + 1)), st
    )
end

struct MultiScaleSkipDeepEquilibriumNetwork{N,L,M,Sh,A,S,K} <: AbstractSkipDeepEquilibriumNetwork
    model::M
    shortcut::Sh
    solver::A
    sensealg::S
    scales::NTuple{N,NTuple{L,Int64}}
    kwargs::K
end

function initialstates(rng::AbstractRNG, deq::MultiScaleSkipDeepEquilibriumNetwork)
    return (
        model=initialstates(rng, deq.model),
        shortcut=initialstates(rng, deq.shortcut),
        split_idxs=Tuple(vcat(0, cumsum(prod.(deq.scales))...)),
        fixed_depth=0,
    )
end

function MultiScaleSkipDeepEquilibriumNetwork(
    main_layers::Tuple,
    mapping_layers::Matrix,
    post_fuse_layer::Union{Nothing,Tuple},
    shortcut_layers::Union{Nothing,Tuple},
    solver,
    scales;
    sensealg=SteadyStateAdjoint(0.1f0, 0.1f0, 10),
    kwargs...,
)
    l1 = ExplicitFluxLayers.Parallel(nothing, main_layers...)
    l2 = ExplicitFluxLayers.BranchLayer(
        ExplicitFluxLayers.Parallel.(+, map(x -> tuple(x...), eachrow(mapping_layers))...)...
    )
    model = if post_fuse_layer === nothing
        ExplicitFluxLayers.Chain(l1, l2)
    else
        l3 = ExplicitFluxLayers.Parallel(nothing, post_fuse_layer...)
        ExplicitFluxLayers.Chain(l1, l2, l3)
    end
    shortcut = if shortcut_layers === nothing
        nothing
    else
        ExplicitFluxLayers.Parallel(nothing, shortcut_layers...)
    end
    return MultiScaleSkipDeepEquilibriumNetwork(model, shortcut, solver, sensealg, scales, kwargs)
end

function (deq::MultiScaleSkipDeepEquilibriumNetwork{N,L,M,Sh})(
    x::AbstractArray{T}, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple
) where {N,L,M,Sh,T}
    z, st = if Sh == Nothing
        u0, st_ = get_initial_condition_mdeq(deq.scales, x, st)
        u0_ = split_and_reshape(u0, st.split_idxs, deq.scales)
        z0, st__ = deq.model(((u0_[1], x), u0_[2:N]...), ps.model, st_.model)
        @set! st_.model = st__
        (vcat(Flux.flatten.(z0)...), st_)
    else
        z0, st_ = deq.shortcut(x, ps.shortcut, st.shortcut)
        @set! st.shortcut = st_
        (vcat(Flux.flatten.(z0)...), st)
    end

    if !iszero(st.fixed_depth)
        z_star = split_and_reshape(z, st.split_idxs, deq.scales)
        st_ = st.model

        for _ in 1:(st.fixed_depth)
            z_star, st_ = deq.model(((z_star[1], x), z_star[2:N]...), ps.model, st_)
        end

        @set! st.model = st_

        return (z_star, DeepEquilibriumSolution(vcat(Flux.flatten.(z_star)...), z, z, 0.0f0, st.fixed_depth)), st
    end

    function dudt_(u, p, t)
        u_split = split_and_reshape(u, st.split_idxs, deq.scales)
        u_, st_ = deq.model(((u_split[1], x), u_split[2:N]...), p, st.model)
        return u_, st_
    end

    dudt(u, p, t) = vcat(Flux.flatten.(dudt_(u, p, t)[1])...) .- u

    prob = SteadyStateProblem(ODEFunction{false}(dudt), z, ps.model)
    sol = solve(prob, deq.solver; sensealg=deq.sensealg, deq.kwargs...)
    z_star, st_ = dudt_(sol.u, ps.model, nothing)

    residual = dudt(sol.u, ps.model, nothing)

    @set! st.model = st_

    return (
        (z_star, DeepEquilibriumSolution(vcat(Flux.flatten.(z_star)...), z, residual, 0.0f0, sol.destats.nf + 1)), st
    )
end
