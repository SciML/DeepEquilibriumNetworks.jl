__backing(Δ::CRC.Tangent) = __backing(CRC.backing(Δ))
__backing(Δ::Tuple) = __backing.(Δ)
__backing(Δ::NamedTuple{F}) where {F} = NamedTuple{F}(__backing(values(Δ)))
__backing(Δ) = Δ

function CRC.rrule(::Type{<:DeepEquilibriumSolution}, z_star::T, u0::T, residual::T,
    jacobian_loss::R, nfe::Int) where {T, R <: AbstractFloat}
    function deep_equilibrium_solution_pullback(dsol)
        return (∂∅, dsol.z_star, dsol.u0, dsol.residual, dsol.jacobian_loss, dsol.nfe)
    end
    return (DeepEquilibriumSolution(z_star, u0, residual, jacobian_loss, nfe),
        deep_equilibrium_solution_pullback)
end

function _safe_getfield(x::NamedTuple{fields}, field) where {fields}
    return field ∈ fields ? getfield(x, field) : ∂∅
end

function CRC.rrule(::Type{T}, args...) where {T <: NamedTuple}
    y = T(args...)
    function nt_pullback(dy)
        fields = fieldnames(T)
        dy isa CRC.Tangent && (dy = CRC.backing(dy))
        return (∂∅, _safe_getfield.((dy,), fields)...)
    end
    return y, nt_pullback
end

function CRC.rrule(::typeof(Setfield.set), obj, l::Setfield.PropertyLens{field},
    val) where {field}
    res = Setfield.set(obj, l, val)
    function setfield_pullback(Δres)
        Δres = __backing(Δres)
        Δobj = Setfield.set(obj, l, ∂∅)
        return (∂∅, Δobj, ∂∅, getfield(Δres, field))
    end
    return res, setfield_pullback
end

function CRC.rrule(::typeof(_construct_problem), deq::AbstractDEQs, dudt, z,
    ps::NamedTuple{F}, x) where {F}
    prob = _construct_problem(deq, dudt, z, ps, x)
    function ∇_construct_problem(Δ)
        Δ = __backing(Δ)
        nograds = NamedTuple{F}(ntuple(i -> ∂∅, length(F)))
        return (∂∅, ∂∅, ∂∅, Δ.u0, merge(nograds, (; model=Δ.p.ps)), Δ.p.x)
    end
    return prob, ∇_construct_problem
end
