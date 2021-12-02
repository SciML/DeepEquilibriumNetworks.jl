#= Test Case

using FastDEQ, Flux

struct MyCustomModel
    W1
    W2
    W3
end

Flux.trainable(m::MyCustomModel) = (m.W1, m.W2)  # Note m.W3 is not trainable

Flux.@functor MyCustomModel  # All leaf nodes can be moved across devices

model = MyCustomModel(rand(2), rand(2), rand(2))

p1, re1 = Flux.destructure(model)  # `p` should be of length 4 but gives us length 6

@btime Flux.destructure($model)  # 2.212 μs
@btime $re1($p1)                 # 2.803 μs

p2, re2 = FastDEQ.destructure(model)

@btime FastDEQ.destructure($model)  # 2.390 μs
@btime $re2($p2)                    # 2.506 μs

=#

function destructure(m)
    ps = Flux.params(m).params  # IdSet
    xs = Zygote.Buffer([])

    # FIXME: Same Layer if appears twice will give erroneous values
    fmap(m) do x
        if x isa AbstractArray{<:Number} && x ∈ ps
            delete!(ps, x)
            push!(xs, vec(x))
        end
        return x
    end

    return vcat(copy(xs)...), p -> _restructure(m, p)
end

function _restructure(m, xs)
    i = 0
    ps = Flux.params(m).params  # IdSet
    m̄ = fmap(m) do x
        if !(x isa AbstractArray{<:Number}) || x ∉ ps
            return x
        end
        x = reshape(xs[i .+ (1:length(x))], size(x))
        i += length(x)
        delete!(ps, x)
        return x
    end
    length(xs) == i || @warn "Expected $(i) params, got $(length(xs))"
    return m̄
end

Zygote.@adjoint function _restructure(m, xs)
    m̄, numel = _restructure(m, xs), length(xs)
    function _restructure_pullback(dm)
        xs′ = destructure(dm)[1]
        if numel != length(xs′)
            @warn "Expected $(numel) params, got $(length(xs′))"
        end
        return (nothing, xs′)
    end
    return m̄, _restructure_pullback
end
