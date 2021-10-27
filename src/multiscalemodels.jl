struct SingleResolutionFeatures{B} <: AbstractMultiScaleArrayLeaf{B}
    values::Vector{B}
end

struct MultiResolutionFeatures{T<:AbstractMultiScaleArray,B<:Number} <:
       AbstractMultiScaleArrayHead{B}
    nodes::Vector{T}
    values::Vector{B}
    end_idxs::Vector{Int}
end

Zygote.@adjoint function SingleResolutionFeatures(v::Vector)
    res = SingleResolutionFeatures(v)
    function singleres_sensitivity(Δ)
        return (Δ.values,)
    end
    return res, singleres_sensitivity
end

Zygote.@adjoint function MultiScaleArrays.construct(
    ::Type{MultiResolutionFeatures},
    nodes,
    args...,
)
    res = construct(MultiResolutionFeatures, nodes)

    construct_sensitivity(Δ::MultiResolutionFeatures) =
        (nothing, Δ.nodes, [nothing for _ in args]...)

    function construct_sensitivity(Δ::Vector)
        s = Vector{SingleResolutionFeatures}(undef, length(nodes))
        start_idx = 1
        for i = 1:length(nodes)
            s[i] = SingleResolutionFeatures(
                Δ[start_idx:start_idx+length(nodes[i].values)-1],
            )
            start_idx += length(nodes[i].values)
        end
        return (nothing, s, [nothing for _ in args]...)
    end

    return res, construct_sensitivity
end

Base.vec(v::MultiResolutionFeatures) = v[:]

# Probably should not do this
function Base.vec(v::NamedTuple{(:nodes, :values, :end_idxs), Tuple{Vector{Tangent{Any, NamedTuple{(:values,), Tuple{Vector{Float64}}}}}, Nothing, Nothing}})
    return vcat([x.values for x in v.nodes]...)
end

function Base.vec(v::NamedTuple{(:nodes, :values, :end_idxs), Tuple{Vector{Tangent{Any, NamedTuple{(:values,), Tuple{Vector{Float32}}}}}, Nothing, Nothing}})
    return vcat([x.values for x in v.nodes]...)
end

struct Depth2MDEQ
    re1::Any
    p1::Any
    re2::Any
    p2::Any
    re12::Any
    p12::Any
    re21::Any
    p21::Any
end

Flux.@functor Depth2MDEQ

function Depth2MDEQ(model1, model2, model12, model21)
    p1, re1 = Flux.destructure(model1)
    p2, re2 = Flux.destructure(model2)
    p12, re12 = Flux.destructure(model12)
    p21, re21 = Flux.destructure(model21)
    return Depth2MDEQ(re1, p1, re2, p2, re12, p12, re21, p21)
end

function (d2mdeq::Depth2MDEQ)(x)
    u0_1 = zero(x)
    u0_2 = d2mdeq.re12(d2mdeq.p12)(u0_1)
    dims1 = size(u0_1)
    dims2 = size(u0_2)
    u0 = construct(
        MultiResolutionFeatures,
        [
            SingleResolutionFeatures(vec(u0_1)),
            SingleResolutionFeatures(vec(u0_2)),
        ],
    )

    p1l = length(d2mdeq.p1)
    p2l = length(d2mdeq.p2)
    p12l = length(d2mdeq.p12)
    p21l = length(d2mdeq.p21)

    function dudt(u, _p, t)
        u1_prev = reshape(u.nodes[1].values, dims1)
        u2_prev = reshape(u.nodes[2].values, dims2)

        _z1 = d2mdeq.re1(_p[1:p1l])(u1_prev .+ x)
        _z2 = d2mdeq.re2(_p[p1l+1:p1l+p2l])(u2_prev)

        u1 = _z1 .+ d2mdeq.re21(_p[p1l+p2l+p12l+1:p1l+p2l+p12l+p21l])(_z2)
        u2 = _z2 .+ d2mdeq.re12(_p[p1l+p2l+1:p1l+p2l+p12l])(_z1)

        return construct(
            MultiResolutionFeatures,
            [
                SingleResolutionFeatures(vec(u1 .- u1_prev)),
                SingleResolutionFeatures(vec(u2 .- u2_prev)),
            ],
        )
    end

    ssprob = SteadyStateProblem(
        dudt,
        u0,
        vcat(d2mdeq.p1, d2mdeq.p2, d2mdeq.p12, d2mdeq.p21),
    )
    sol = solve(
        ssprob,
        DynamicSS(
            Tsit5();
            abstol = 1.0f0,
            reltol = 1.0f0,
            tspan = (0.0f0, Inf32),
        );
        u0 = u0,
        sensealg = SteadyStateAdjoint(
            autodiff = false,
            autojacvec = ZygoteVJP(),
            linsolve = LinSolveKrylovJL(rtol = 0.1, atol = 0.1),
        ),
        reltol = 1.0f0,
        abstol = 1.0f0,
    ).u
    return (reshape(sol.nodes[1].values, dims1), reshape(sol.nodes[2].values, dims2))
end

# # res1 = SingleResolutionFeatures(rand(4))
# # res2 = SingleResolutionFeatures(rand(8))