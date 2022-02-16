"""
    MultiParallelNet(layers...)
    MultiParallelNet(layers::Tuple)
    MultiParallelNet(layers::Vector)

Creates a MultiParallelNet mostly used for MultiScale Models. It takes a list of inputs
and passes all of them through each `layer` and returns a tuple of outputs.

## Example

```
Model := MultiParallelNet(L1, L2, L3)

Model(X1, X2) := (Model.L1(X1, X2), Model.L2(X1, X2), Model.L3(X1, X2))
```
"""
struct MultiParallelNet{L}
    layers::L

    function MultiParallelNet(args...)
        layers = tuple(args...)
        return new{typeof(layers)}(layers)
    end

    MultiParallelNet(layers::Tuple) = new{typeof(layers)}(layers)

    MultiParallelNet(layers::Vector) = MultiParallelNet(layers...)
end

Flux.@functor MultiParallelNet

function (mpn::MultiParallelNet)(x::Union{Tuple,Vector})
    buf = Zygote.Buffer(Vector{Any}(undef, length(mpn.layers)))
    for (i, l) in enumerate(mpn.layers)
        buf[i] = l(x...)
    end
    return Tuple(copy(buf))
end

function (mpn::MultiParallelNet)(args...)
    buf = Zygote.Buffer(Vector{Any}(undef, length(mpn.layers)))
    for (i, l) in enumerate(mpn.layers)
        buf[i] = l(args...)
    end
    return Tuple(copy(buf))
end
