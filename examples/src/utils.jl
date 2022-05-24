# unsafe_free OneHotArrays
CUDA.unsafe_free!(x::OneHotArray) = CUDA.unsafe_free!(x.indices)

# Memory Management
relieve_gc_pressure(::Union{Nothing,<:AbstractArray}) = nothing
relieve_gc_pressure(x::CuArray) = CUDA.unsafe_free!(x)
relieve_gc_pressure(t::Tuple) = relieve_gc_pressure.(t)
relieve_gc_pressure(x::NamedTuple) = fmap(relieve_gc_pressure, x)

function invoke_gc()
    GC.gc(true)
    CUDA.reclaim()
    return nothing
end

# Optimisers / Parameter Schedulers
function update_lr(st::ST, eta) where {ST}
    if hasfield(ST, :eta)
        @set! st.eta = eta
    end
    return st
end
update_lr(st::Optimisers.OptimiserChain, eta) = update_lr.(st.opts, eta)
function update_lr(st::Optimisers.Leaf, eta)
    @set! st.rule = update_lr(st.rule, eta)
end
update_lr(st_opt::NamedTuple, eta) = fmap(l -> update_lr(l, eta), st_opt)

# Metrics
accuracy(ŷ, y) = sum(argmax.(eachcol(ŷ)) .== onecold(y)) * 100 / size(y, ndims(y))

function accuracy(ŷ, y, topk::NTuple{N,<:Int}) where {N}
    maxk = maximum(topk)

    pred_labels = partialsortperm.(eachcol(ŷ), (1:maxk,), rev=true)
    true_labels = onecold(y)

    accuracies = Tuple(sum(map((a, b) -> sum(view(a, 1:k) .== b), pred_labels, true_labels)) for k in topk)

    return accuracies .* 100 ./ size(y, ndims(y))
end

# Distributed Utils
@inline is_distributed() = FluxMPI.Initialized() && total_workers() > 1
@inline should_log() = !FluxMPI.Initialized() || local_rank() == 0
@inline scaling_factor() = (is_distributed() ? total_workers() : 1)

# Loss Function
@inline logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims=1); dims=1))
@inline mae(ŷ, y) = mean(abs, ŷ .- y)
@inline mse(ŷ, y) = mean(abs2, ŷ .- y)

# DataLoaders doesn't yet work with MLUtils
MLDataPattern.nobs(x) = MLUtils.numobs(x)
MLDataPattern.getobs(d::MLUtils.ObsView, i::Int64) = MLUtils.getobs(d, i)
