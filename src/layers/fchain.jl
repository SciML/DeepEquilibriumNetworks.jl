# MDEQ takes around 45 mins to compile for the first run
# https://github.com/FluxML/Zygote.jl/issues/1126
struct FChain{T}
    layers::T
    function FChain(xs...)
        @show 1, xs
        xs = flatten_model(xs)
        @show 2, xs
        return new{typeof(xs)}(xs)
    end
    FChain(xs::AbstractVector) = new{typeof(xs)}(xs)  # allows FChain(Any[Layer, Layer, ...]), type-unstable
end

function flatten_model(layers::Union{AbstractVector,Tuple})
    new_layers = []
    for l in layers
        f = flatten_model(l)
        if f isa Tuple || f isa AbstractVector
            append!(new_layers, f)
        elseif f isa FChain
            append!(new_layers, f.layers)
        elseif f isa Chain
            append!(new_layers, f.layers...)
        else
            push!(new_layers, f)
        end
    end
    return layers isa AbstractVector ? new_layers : Tuple(new_layers)
end

flatten_model(x) = x

flatten_model(x::Chain) = Chain(flatten_model(x.layers))

Flux.@forward FChain.layers Base.getindex, Base.length, Base.first, Base.last, Base.iterate, Base.lastindex, Base.keys

@functor FChain

# (c::FChain)(x) = foldl((y,f) -> f(y), c.layers; init=x) # NO, this forgets the gradient for x

(c::FChain)(x) = foldl((y, f) -> f(y), (x, c.layers...))

(c::FChain)(x...) = foldl((y, f) -> f(y...), (x, c.layers...))

(c::FChain{<:AbstractVector})(x) = foldl((y, f) -> f(y), vcat([x], c.layers))

(c::FChain{<:AbstractVector})(x...) = foldl((y, f) -> f(y...), vcat([x], c.layers))

Base.getindex(c::FChain, i::AbstractArray) = FChain(c.layers[i]...)

function Base.show(io::IO, c::FChain)
    print(io, "FChain(")
    Flux._show_layers(io, c.layers)
    print(io, ")")
end

Flux._show_layers(io, layers::AbstractVector) = join(io, layers, ", ")
