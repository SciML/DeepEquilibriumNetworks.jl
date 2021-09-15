struct DEQFixedPointLayer{F,S}
    f::F
    solver::S
end

function DEQFixedPointLayer(f, dummy_x::AbstractArray; kwargs...)
    init_z = zero(dummy_x)
    solver = AndersonAcceleration(init_z; kwargs...)
    return DEQFixedPointLayer(f, solver)
end

Flux.@functor DEQFixedPointLayer

function (deq::DEQFixedPointLayer)(x)
    init_z = zero(x)

    nn(z) = deq.f(z, x)

    z₀ = Zygote.@ignore fixedpointsolve(deq.solver, nn, init_z)

    return Zygote.hook(grad -> deq_backward_hook(deq, z₀, x, grad), deq.f(z₀, x))
end

function deq_backward_hook(deq::DEQFixedPointLayer, z₀, x, grad)
    function f(y)
        _, back = Zygote.pullback(z -> deq.f(z, x), z₀)
        return back(y)[1] .+ grad
    end

    return fixedpointsolve(deq.solver, f, grad)
end
