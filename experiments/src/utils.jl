import CUDA, FluxMPI, Functors, JLSO, OneHotArrays, Setfield, Statistics, Zygote
import Lux.Training

"""
    flatten_tree(tree; delim::String=".")

Flatten a nested NamedTuple to a "flat" NamedTuple by joining the keys with `delim`.
"""
flatten_tree(node; delim::String=".") = node  # Anything not a NamedTuple is a Node

function flatten_tree(tree::NamedTuple{fields}; delim::String=".") where {fields}
  children = flatten_tree.(values(tree); delim)
  children_names = collect(keys(tree))
  flattened_tree = []
  for (name, val) in zip(children_names, children)
    if !isa(val, NamedTuple)
      push!(flattened_tree, name => val)
    else
      for (k, v) in pairs(val)
        full_name = Symbol(join((name, k), delim))
        push!(flattened_tree, full_name => v)
      end
    end
  end
  return NamedTuple(flattened_tree)
end

"""
    print_tree(tree; indent::Int=0, indent_increment::Int=4)

Pretty Print a nested NamedTuple.
"""
function print_tree(tree::NamedTuple; indent::Int=0, indent_increment::Int=4)
  for (k, v) in pairs(tree)
    if !isa(v, NamedTuple)
      if v != ""
        println(" "^indent, k, " => ", v)
      else
        println(" "^indent, k, " => \"\"")
      end
    else
      println(" "^indent, k)
      print_tree(v; indent=indent + indent_increment, indent_increment)
    end
  end
  return nothing
end

"""
    accuracy(y_pred::AbstractMatrix, y::AbstractMatrix)
    accuracy(y_pred::AbstractMatrix, y::AbstractMatrix, topk::NTuple{N,<:Int})

Compute the percentage of matches between y_pred and y.
"""
function accuracy(y_pred::AbstractMatrix, y::AbstractMatrix)
  return sum(argmax.(eachcol(y_pred)) .== OneHotArrays.onecold(y)) * 100 / size(y, 2)
end

function accuracy(y_pred::AbstractMatrix, y::AbstractMatrix,
                  topk::NTuple{N, <:Int}) where {N}
  maxk = maximum(topk)

  pred_labels = partialsortperm.(eachcol(y_pred), (1:maxk,), rev=true)
  true_labels = OneHotArrays.onecold(y)

  accuracies = Tuple(sum(map((a, b) -> sum(view(a, 1:k) .== b), pred_labels, true_labels))
                     for k in topk)

  return accuracies .* 100 ./ size(y, ndims(y))
end

function logitcrossentropy(y_pred, y; dims=1)
  return Statistics.mean(-sum(y .* NNlib.logsoftmax(y_pred; dims=dims); dims=dims))
end

function mean_absolute_error(y_pred, y)
  return Statistics.mean(abs, y_pred .- y)
end

function mean_squared_error(y_pred, y)
  return Statistics.mean(abs2, y_pred .- y)
end

# Parameter Scheduling
## Copied from ParameterSchedulers.jl due to its heavy dependencies
struct CosineAnnealSchedule{restart, T, S <: Integer}
  range::T
  offset::T
  dampen::T
  period::S

  function CosineAnnealSchedule(lambda_0, lambda_1, period; restart::Bool=true,
                                dampen=1.0f0)
    range = abs(lambda_0 - lambda_1)
    offset = min(lambda_0, lambda_1)
    return new{restart, typeof(range), typeof(period)}(range, offset, dampen, period)
  end
end

function (s::CosineAnnealSchedule{true})(t)
  d = s.dampen^div(t - 1, s.period)
  return (s.range * (1 + cos(pi * mod(t - 1, s.period) / s.period)) / 2 + s.offset) / d
end

function (s::CosineAnnealSchedule{false})(t)
  return s.range * (1 + cos(pi * (t - 1) / s.period)) / 2 + s.offset
end

struct ConstantSchedule{T}
  val::T
end

(s::ConstantSchedule)(t) = s.val

struct Step{T, S}
  start::T
  decay::T
  step_sizes::S

  function Step(start::T, decay::T, step_sizes::S) where {T, S}
    _step_sizes = (S <: Integer) ? Iterators.repeated(step_sizes) : step_sizes

    return new{T, typeof(_step_sizes)}(start, decay, _step_sizes)
  end
end

(s::Step)(t) = s.start * s.decay^(searchsortedfirst(s.step_sizes, t - 1) - 1)

# Generic Stuff
is_distributed() = FluxMPI.Initialized() && FluxMPI.total_workers() > 1

should_log() = !is_distributed() || FluxMPI.local_rank() == 1

## Memory Management
relieve_gc_pressure(::Union{Nothing, <:AbstractArray}) = nothing
relieve_gc_pressure(x::CUDA.CuArray) = CUDA.unsafe_free!(x)
relieve_gc_pressure(t::Tuple) = relieve_gc_pressure.(t)
relieve_gc_pressure(x::NamedTuple) = fmap(relieve_gc_pressure, x)

function invoke_gc()
  GC.gc(true)
  CUDA.reclaim()
  return nothing
end

function warmup_model(loss_function, model, ps, st, cfg::ExperimentConfig; transform_input)
  x = ones(Float32, cfg.model.image_size..., cfg.model.in_channels, 1) |> transform_input
  y = OneHotArrays.onehotbatch([1], 0:(cfg.model.num_classes - 1)) |> transform_input

  should_log() && @info "model warmup started"
  loss_function(model, ps, st, (x, y))
  should_log() && @info "forward pass warmup completed"
  Zygote.gradient(p -> first(loss_function(model, ps, st, (x, y))), ps)
  should_log() && @info "backward pass warmup completed"
  st_ = Lux.update_state(st, :fixed_depth, Val(10))
  loss_function(model, ps, st_, (x, y))
  should_log() && @info "forward pass (pretraining) warmup completed"
  Zygote.gradient(p -> first(loss_function(model, ps, st_, (x, y))), ps)
  should_log() && @info "backward pass (pretraining) warmup completed"

  return nothing
end

function save_checkpoint(state::NamedTuple; is_best::Bool, filename::String)
  isdir(dirname(filename)) || mkpath(dirname(filename))
  JLSO.save(filename, :state => state)
  is_best && _symlink_safe(filename, joinpath(dirname(filename), "model_best.jlso"))
  _symlink_safe(filename, joinpath(dirname(filename), "model_current.jlso"))
  return nothing
end

function _symlink_safe(src, dest)
  rm(dest; force=true)
  return symlink(src, dest)
end

function load_checkpoint(fname::String)
  try
    # NOTE(@avik-pal): ispath is failing for symlinks?
    return JLSO.load(fname)[:state]
  catch
    @warn """$fname could not be loaded. This might be because the file is absent or is
             corrupt. Proceeding by returning `nothing`."""
    return nothing
  end
end

function run_training_step(::Training.ZygoteVJP, objective_function, data,
                           ts::Training.TrainState)
  t = time()
  (loss, st, stats), back = Zygote.pullback(ps -> objective_function(ts.model, ps,
                                                                     ts.states, data),
                                            ts.parameters)
  fwd_time = time() - t

  t = time()
  grads = back((one(loss), nothing, nothing))[1]
  if is_distributed()
    grads = FluxMPI.allreduce_gradients(grads)
  end
  bwd_time = time() - t

  Setfield.@set! ts.states = st
  t = time()
  ts = Training.apply_gradients(ts, grads)
  opt_time = time() - t

  return loss, st, stats, ts, grads, (; fwd_time, bwd_time, opt_time)
end

function FluxMPI.synchronize!(tstate::Training.TrainState; root_rank::Int=0)
  Setfield.@set! tstate.parameters = FluxMPI.synchronize!(tstate.parameters; root_rank)
  Setfield.@set! tstate.states = FluxMPI.synchronize!(tstate.states; root_rank)
  Setfield.@set! tstate.optimizer_state = FluxMPI.synchronize!(tstate.optimizer_state;
                                                               root_rank)
  return tstate
end
