function CRC.rrule(::Type{<:DeepEquilibriumSolution}, z_star::T, u0::T, residual::T,
    jacobian_loss::R, nfe::Int) where {T, R <: AbstractFloat}
    function deep_equilibrium_solution_pullback(dsol)
        return (CRC.NoTangent(), dsol.z_star, dsol.u0, dsol.residual, dsol.jacobian_loss,
            dsol.nfe)
    end
    return (DeepEquilibriumSolution(z_star, u0, residual, jacobian_loss, nfe),
        deep_equilibrium_solution_pullback)
end

function _safe_getfield(x::NamedTuple{fields}, field) where {fields}
    return field ∈ fields ? getfield(x, field) : CRC.NoTangent()
end

function CRC.rrule(::Type{T}, args...) where {T <: NamedTuple}
    y = T(args...)
    function nt_pullback(dy)
        fields = fieldnames(T)
        if dy isa CRC.Tangent
            dy = CRC.backing(dy)
        end
        return (CRC.NoTangent(), _safe_getfield.((dy,), fields)...)
    end
    return y, nt_pullback
end

function CRC.rrule(::typeof(Setfield.set), obj, l::Setfield.PropertyLens{field},
    val) where {field}
    res = Setfield.set(obj, l, val)
    function setfield_pullback(Δres)
        if Δres isa CRC.Tangent
            Δres = CRC.backing(Δres)
        end
        Δobj = Setfield.set(obj, l, CRC.NoTangent())
        return (CRC.NoTangent(), Δobj, CRC.NoTangent(), getfield(Δres, field))
    end
    return res, setfield_pullback
end

# Honestly no clue why this is needed! -- probably a whacky fix which shouldn't be ever
# needed.
ZygoteRules.gradtuple1(::NamedTuple{()}) = (nothing, nothing, nothing, nothing, nothing)
ZygoteRules.gradtuple1(x::NamedTuple) = collect(values(x))
