Zygote.@adjoint function SingleResolutionFeatures(values)
    srf = SingleResolutionFeatures(values)
    function srf_sensitivity(Δ)
        return (Δ.values,)
    end
    return srf, srf_sensitivity
end

Zygote.@adjoint function Zygote.literal_getproperty(
    srf::SingleResolutionFeatures,
    ::Val{:values},
)
    return (
        getproperty(srf, :values),
        Δ -> (SingleResolutionFeatures(Δ), nothing)
    )
end

Zygote.@adjoint function Zygote.literal_getproperty(
    mrf::MultiResolutionFeatures{T},
    ::Val{:nodes},
) where {T}
    nodes = getproperty(mrf, :nodes)
    return (
        nodes,
        Δ -> begin
            Δ_corrected = Vector{T}(undef, length(nodes))
            for (i, _Δ) in enumerate(Δ)
                if _Δ isa NoTangent
                    Δ_corrected[i] = zero(nodes[i])
                else
                    Δ_corrected[i] = _Δ
                end
            end
            return (
                MultiResolutionFeatures(
                    Δ_corrected,
                    zero(getproperty(mrf, :values)),
                    zero(getproperty(mrf, :end_idxs)),
                ),
                nothing,
            )
        end,
    )
end

Zygote.@adjoint function Zygote.literal_getproperty(
    mrf::MultiResolutionFeatures,
    ::Val{:values},
)
    return (
        getproperty(mrf, :values),
        Δ -> (
            MultiResolutionFeatures(
                zero.(getproperty(mrf, :nodes)),
                Δ,
                zero(getproperty(mrf, :end_idxs)),
            ),
            nothing,
        ),
    )
end

Zygote.@adjoint function Zygote.literal_getproperty(
    mrf::MultiResolutionFeatures,
    ::Val{:end_idxs},
)
    return (
        getproperty(mrf, :end_idxs),
        Δ -> (
            MultiResolutionFeatures(
                zero.(getproperty(mrf, :nodes)),
                zero(getproperty(mrf, :values)),
                Δ,
            ),
            nothing,
        ),
    )
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

Zygote.@adjoint function Zygote.literal_getproperty(
    sol::SciMLBase.NonlinearSolution,
    ::Val{:u},
)
    function solu_adjoint(Δ::MultiResolutionFeatures)
        (DiffEqBase.build_solution(sol.prob, sol.alg, Δ, sol.resid),)
    end
    function solu_adjoint(Δ)
        zerou = zero(sol.prob.u0)
        _Δ = @. ifelse(Δ === nothing, zerou, Δ)
        (DiffEqBase.build_solution(sol.prob, sol.alg, _Δ, sol.resid),)
    end
    sol.u, solu_adjoint
end