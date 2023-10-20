function CRC.rrule(::Type{<:DeepEquilibriumSolution}, z_star::T, u0::T, residual::T,
    jacobian_loss::R, nfe::Int) where {T, R <: AbstractFloat}
    function ∇deep_equilibrium_solution(dsol)
        return (∂∅, dsol.z_star, dsol.u0, dsol.residual, dsol.jacobian_loss, dsol.nfe)
    end
    return (DeepEquilibriumSolution(z_star, u0, residual, jacobian_loss, nfe),
        ∇deep_equilibrium_solution)
end
