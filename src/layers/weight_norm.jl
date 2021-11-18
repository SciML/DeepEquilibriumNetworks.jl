struct WeightNorm{Re,P,D}
    layer_re::Re
    parameters::P
    dims::D
end

Flux.@functor WeightNorm (parameters,)

function Base.show(io::IO, wn::WeightNorm)
    p = get_updated_parameters(wn)
    l = wn.layer_re(p)
    print(io, "WeightNorm(")
    print(io, l)
    print(") ", string(length(p)), " Trainable Parameters")
end

function WeightNorm(layer, dim::Union{Tuple,Vector,Int})
    ps = Flux.params(layer)
    dim isa Int && (dim = [dim for _ = 1:length(ps)])

    _, layer_re = Flux.destructure(layer)

    parameters = []
    for (i, p) in enumerate(ps)
        g_val = _norm(p, dim[i])
        v_val = copy(p)
        push!(parameters, (g_val, v_val))
    end

    return WeightNorm(layer_re, tuple(parameters...), dim)
end

compute_normed_weight(v, g, dim) = v .* (g ./ _norm(v, dim))

get_updated_parameters(wn::WeightNorm) = vcat(
    ntuple(
        i -> vec(
            compute_normed_weight(
                wn.parameters[i][2],
                wn.parameters[i][1],
                wn.dims[i],
            ),
        ),
        length(wn.dims),
    )...,
)

function (wn::WeightNorm)(args...; kwargs...)
    # TODO: We should not have to update when inside DEQ
    p = get_updated_parameters(wn)
    return wn.layer_re(p)(args...; kwargs...)
end
