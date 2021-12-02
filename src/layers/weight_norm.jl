struct WeightNorm{Re,P,D}
    layer_re::Re
    parameters::P
    dims::D
end

Flux.@functor WeightNorm (parameters,)

function Base.show(io::IO, wn::WeightNorm)
    ps = sum(length.(Flux.params(wn)))
    p = update_parameters(wn)
    l = wn.layer_re(p)
    print(io, "WeightNorm(")
    print(io, l)
    return print(") ", string(ps), " Trainable Parameters")
end

function WeightNorm(layer, dim::Union{Tuple,Vector,Int,Nothing}=nothing)
    ps = Flux.params(layer)
    dim = dim === nothing ? [ndims(p) for p in ps] : (dim isa Int ? [dim for _ in 1:length(ps)] : dim)

    p_, layer_re = destructure(layer)

    parameters = []
    for (i, p) in enumerate(ps)
        g_val = _norm(p, dim[i])
        v_val = copy(p)
        push!(parameters, (g_val, v_val))
    end

    return WeightNorm(layer_re, tuple(parameters...), dim)
end

compute_normed_weight(v, g, dim) = v .* (g ./ _norm(v, dim))

function update_parameters(wn::WeightNorm)
    return vcat(ntuple(i -> vec(compute_normed_weight(wn.parameters[i][2], wn.parameters[i][1], wn.dims[i])),
                       length(wn.dims))...)
end

function (wn::WeightNorm)(args...; kwargs...)
    return wn.layer_re(update_parameters(wn))(args...; kwargs...)
end
