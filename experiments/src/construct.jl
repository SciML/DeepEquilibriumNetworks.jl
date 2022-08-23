import Lux, NNlib, Optimisers, Random
import DeepEquilibriumNetworks as DEQ

function normal_initializer(rng::Random.AbstractRNG, dims...; mean::T=0.0f0,
                            std::T=0.01f0) where {T}
  return randn(rng, T, dims...) .* std .+ mean
end

addrelu(x, y) = NNlib.relu.(x .+ y)

reassociate((x1, x2), y) = (x1, (x2, y))

addtuple((x, y)) = x .+ y

function compute_feature_scales(; image_size, downsample_times, num_channels, num_branches)
  image_size_downsampled = div.(image_size, 2^downsample_times)
  scales = [(image_size_downsampled..., num_channels[1])]
  for i in 2:num_branches
    push!(scales, ((scales[end][1:2] .÷ 2)..., num_channels[i]))
  end
  return Tuple(scales)
end

# Building Blocks
function conv1x1(mapping, activation=identity; stride::Int=1, use_bias=false, dilation=1,
                 groups=1)
  return Lux.Conv((1, 1), mapping, activation; pad=0, init_weight=normal_initializer,
                  stride, use_bias, dilation, groups)
end

function conv3x3(mapping, activation=identity; stride::Int=1, use_bias=false, dilation=1,
                 groups=1)
  return Lux.Conv((3, 3), mapping, activation; pad=1, init_weight=normal_initializer,
                  stride, use_bias, dilation, groups)
end

function conv5x5(mapping, activation=identity; stride::Int=1, use_bias=false, dilation=1,
                 groups=1)
  return Lux.Conv((5, 5), mapping, activation; pad=2, init_weight=normal_initializer,
                  stride, use_bias, dilation, groups)
end

function downsample_module(mapping, level_difference, activation; group_count=8)
  in_channels, out_channels = mapping

  function intermediate_mapping(i)
    if in_channels * (2^level_difference) == out_channels
      return (in_channels * (2^(i - 1))) => (in_channels * (2^i))
    else
      return i == level_difference ? in_channels => out_channels :
             in_channels => in_channels
    end
  end

  layers = Lux.AbstractExplicitLayer[]
  for i in 1:level_difference
    in_channels, out_channels = intermediate_mapping(i)
    push!(layers,
          Lux.Chain(conv3x3(in_channels => out_channels; stride=2),
                    Lux.BatchNorm(out_channels, activation; affine=true,
                                  track_stats=false)))
  end
  return Lux.Chain(layers...; disable_optimizations=true)
end

function upsample_module(mapping, level_difference, activation; group_count=8,
                         upsample_mode::Symbol=:nearest)
  in_channels, out_channels = mapping

  function intermediate_mapping(i)
    if out_channels * (2^level_difference) == in_channels
      (in_channels ÷ (2^(i - 1))) => (in_channels ÷ (2^i))
    else
      i == level_difference ? in_channels => out_channels : in_channels => in_channels
    end
  end

  layers = Lux.AbstractExplicitLayer[]
  for i in 1:level_difference
    in_channels, out_channels = intermediate_mapping(i)
    push!(layers,
          Lux.Chain(conv3x3(in_channels => out_channels),
                    Lux.BatchNorm(out_channels, activation; affine=true, track_stats=false),
                    Lux.Upsample(upsample_mode; scale=2)))
  end
  return Lux.Chain(layers...; disable_optimizations=true)
end

struct ResidualBlock{C1, C2, Dr, Do, N1, N2, N3} <:
       Lux.AbstractExplicitContainerLayer{(:conv1, :conv2, :dropout, :downsample, :norm1,
                                           :norm2, :norm3)}
  conv1::C1
  conv2::C2
  dropout::Dr
  downsample::Do
  norm1::N1
  norm2::N2
  norm3::N3
end

function ResidualBlock(mapping; deq_expand::Int=5, num_gn_groups::Int=4,
                       downsample=Lux.NoOpLayer(), n_big_kernels::Int=0,
                       dropout_rate::Real=0.0f0, gn_affine::Bool=true,
                       weight_norm::Bool=true)
  in_planes, out_planes = mapping
  inner_planes = out_planes * deq_expand
  conv1 = (n_big_kernels >= 1 ? conv5x5 : conv3x3)(in_planes => inner_planes;
                                                   use_bias=false)
  conv2 = (n_big_kernels >= 2 ? conv5x5 : conv3x3)(inner_planes => out_planes;
                                                   use_bias=false)

  if weight_norm
    conv1 = Lux.WeightNorm(conv1, (:weight,), (4,))
    conv2 = Lux.WeightNorm(conv2, (:weight,), (4,))
  end

  norm1 = Lux.BatchNorm(inner_planes, Lux.relu; affine=gn_affine, track_stats=false)
  norm2 = Lux.BatchNorm(out_planes; affine=gn_affine, track_stats=false)
  norm3 = Lux.BatchNorm(out_planes; affine=gn_affine, track_stats=false)

  dropout = Lux.VariationalHiddenDropout(dropout_rate)

  return ResidualBlock(conv1, conv2, dropout, downsample, norm1, norm2, norm3)
end

function (rb::ResidualBlock)((x, y)::Tuple, ps, st)
  x, st_conv1 = rb.conv1(x, ps.conv1, st.conv1)
  x, st_norm1 = rb.norm1(x, ps.norm1, st.norm1)
  x, st_conv2 = rb.conv2(x, ps.conv2, st.conv2)

  x_do, st_downsample = rb.downsample(x, ps.downsample, st.downsample)
  x_dr, st_dropout = rb.dropout(x, ps.dropout, st.dropout)

  y = x_dr .+ y
  y, st_norm2 = rb.norm2(y, ps.norm2, st.norm2)

  y = addrelu(y, x_do)
  y, st_norm3 = rb.norm3(y, ps.norm3, st.norm3)

  return (y,
          (conv1=st_conv1, conv2=st_conv2, dropout=st_dropout, downsample=st_downsample,
           norm1=st_norm1, norm2=st_norm2, norm3=st_norm3))
end

(rb::ResidualBlock)(x::AbstractArray, ps, st) = rb((x, eltype(x)(0)), ps, st)

struct BottleneckBlock{R, C, M} <:
       Lux.AbstractExplicitContainerLayer{(:rescale, :conv, :mapping)}
  rescale::R
  conv::C
  mapping::M
end

function BottleneckBlock(mapping::Pair; expansion::Int=4, bn_track_stats::Bool=true,
                         bn_affine::Bool=true)
  rescale = if first(mapping) != last(mapping) * expansion
    Lux.Chain(conv1x1(first(mapping) => last(mapping) * expansion),
              Lux.BatchNorm(last(mapping) * expansion; affine=bn_affine,
                            track_stats=bn_track_stats))
  else
    Lux.NoOpLayer()
  end

  return BottleneckBlock(rescale, conv1x1(mapping),
                         Lux.Chain(Lux.BatchNorm(last(mapping), NNlib.relu;
                                                 affine=bn_affine,
                                                 track_stats=bn_track_stats),
                                   conv3x3(last(mapping) => last(mapping)),
                                   Lux.BatchNorm(last(mapping), NNlib.relu;
                                                 affine=bn_affine,
                                                 track_stats=bn_track_stats),
                                   conv1x1(last(mapping) => last(mapping) * expansion),
                                   Lux.BatchNorm(last(mapping) * expansion;
                                                 affine=bn_affine,
                                                 track_stats=bn_track_stats);
                                   disable_optimizations=true))
end

function (bn::BottleneckBlock)((x, y)::Tuple, ps, st)
  x_r, st_rescale = bn.rescale(x, ps.rescale, st.rescale)
  x_m, st_conv1 = bn.conv(x, ps.conv, st.conv)
  x_m, st_mapping = bn.mapping(y .+ x_m, ps.mapping, st.mapping)

  return (addrelu(x_m, x_r), (rescale=st_rescale, conv=st_conv1, mapping=st_mapping))
end

(bn::BottleneckBlock)(x::AbstractArray, ps, st) = bn((x, eltype(x)(0)), ps, st)

function get_model(; num_channels, downsample_times, num_branches, expansion_factor,
                   dropout_rate, group_count, big_kernels, head_channels, fuse_method,
                   final_channelsize, num_classes, model_type, solver, sensealg, maxiters,
                   image_size, weight_norm, in_channels)
  init_channel_size = first(num_channels)

  downsample_layers = Lux.AbstractExplicitLayer[]
  for i in 1:(downsample_times + 2)
    stride = i <= 2 ? (downsample_times >= i ? 2 : 1) : 2
    in_channels = i == 1 ? in_channels : init_channel_size
    push!(downsample_layers,
          Lux.Chain(conv3x3(in_channels => init_channel_size; stride),
                    Lux.BatchNorm(init_channel_size, NNlib.relu; affine=true,
                                  track_stats=true)))
  end
  downsample = Lux.Chain(downsample_layers...; disable_optimizations=true)

  if downsample_times == 0 && num_branches <= 2
    stage0 = Lux.NoOpLayer()
  else
    stage0 = Lux.Chain(conv1x1(init_channel_size => init_channel_size),
                       Lux.BatchNorm(init_channel_size, NNlib.relu; affine=true,
                                     track_stats=true))
  end

  initial_layers = Lux.Chain(downsample, stage0; disable_optimizations=true)

  main_layers = Tuple(ResidualBlock(num_channels[i] => num_channels[i];
                                    deq_expand=expansion_factor, dropout_rate=dropout_rate,
                                    num_gn_groups=group_count,
                                    n_big_kernels=big_kernels[i]) for i in 1:(num_branches))

  mapping_layers = Matrix{Lux.AbstractExplicitLayer}(undef, num_branches, num_branches)
  for i in 1:num_branches, j in 1:num_branches
    if i == j
      mapping_layers[i, j] = Lux.NoOpLayer()
    elseif i < j
      mapping_layers[i, j] = downsample_module(num_channels[i] => num_channels[j], j - i,
                                               NNlib.relu; group_count)
    else
      mapping_layers[i, j] = upsample_module(num_channels[i] => num_channels[j], i - j,
                                             NNlib.relu; group_count,
                                             upsample_mode=:nearest)
    end
  end

  post_fuse_layers = Tuple(Lux.Chain(Lux.WrappedFunction(Base.Fix1(broadcast, NNlib.relu)),
                                     conv1x1(num_channels[i] => num_channels[i]),
                                     Lux.BatchNorm(num_channels[i]; affine=true,
                                                   track_stats=false))
                           for i in 1:num_branches)

  increment_modules = Lux.Parallel(nothing,
                                   [BottleneckBlock(num_channels[i] => head_channels[i];
                                                    expansion=4, bn_track_stats=true,
                                                    bn_affine=true) for i in 1:num_branches]...)

  downsample_modules = [Lux.Chain(conv3x3(head_channels[i] * 4 => head_channels[i + 1] * 4;
                                          stride=2, use_bias=true),
                                  Lux.BatchNorm(head_channels[i + 1] * 4, NNlib.relu;
                                                affine=true, track_stats=false))
                        for i in 1:(num_branches - 1)]
  downsample_modules = Lux.PairwiseFusion(fuse_method == "sum" ? (+) :
                                          throw(ArgumentError("unknown `fuse_method` = $(fuse_method)")),
                                          downsample_modules...)

  final_layers = Lux.Chain(increment_modules, downsample_modules,
                           conv1x1(head_channels[num_branches] * 4 => final_channelsize;
                                   use_bias=true),
                           Lux.BatchNorm(final_channelsize, NNlib.relu; affine=true,
                                         track_stats=false), Lux.GlobalMeanPool(),
                           Lux.FlattenLayer(), Lux.Dense(final_channelsize => num_classes);
                           disable_optimizations=true)

  scales = compute_feature_scales(; image_size, downsample_times, num_channels,
                                  num_branches)

  if model_type in ("skip", "skipv2")
    if model_type == "skip"
      shortcut_layers = Lux.AbstractExplicitLayer[]
      push!(shortcut_layers,
            ResidualBlock(num_channels[1] => num_channels[1]; weight_norm=weight_norm))
      for i in 1:(num_branches - 1)
        push!(shortcut_layers,
              downsample_module(num_channels[1] => num_channels[i + 1], i, NNlib.relu;
                                group_count))
      end
      shortcut = tuple(shortcut_layers...)
    else
      shortcut = nothing
    end
    deq = DEQ.MultiScaleSkipDeepEquilibriumNetwork(main_layers, mapping_layers,
                                                   post_fuse_layers, shortcut, solver,
                                                   scales; maxiters=maxiters,
                                                   sensealg=sensealg, verbose=false)
  else
    deq = DEQ.MultiScaleDeepEquilibriumNetwork(main_layers, mapping_layers,
                                               post_fuse_layers, solver, scales;
                                               maxiters=maxiters, sensealg=sensealg,
                                               verbose=false)
  end

  return Lux.Chain(initial_layers, deq, final_layers; disable_optimizations=true)
end

function construct(cfg::ModelConfig)
  return get_model(; cfg.num_channels, cfg.downsample_times, cfg.num_branches,
                   cfg.expansion_factor, cfg.dropout_rate, cfg.group_count, cfg.big_kernels,
                   cfg.head_channels, cfg.fuse_method, cfg.final_channelsize,
                   cfg.num_classes, cfg.model_type, cfg.maxiters, cfg.image_size,
                   cfg.weight_norm, cfg.in_channels, solver=construct(cfg.solver),
                   sensealg=construct(cfg.sensealg))
end

function construct(cfg::SolverConfig)
  if cfg.continuous
    solver = _get_ode_solver(cfg)
    return DEQ.ContinuousDEQSolver(solver; mode=Symbol(cfg.stop_mode), cfg.abstol,
                                   cfg.reltol, cfg.abstol_termination,
                                   cfg.reltol_termination)
  else
    return DEQ.DiscreteDEQSolver(DEQ.LimitedMemoryBroydenSolver();
                                 mode=Symbol(cfg.step_mode), cfg.abstol_termination,
                                 cfg.reltol_termination)
  end
end

function construct(cfg::SensitivityConfig)
  return DEQ.DeepEquilibriumAdjoint(cfg.abstol, cfg.reltol, cfg.maxiters;
                                    mode=cfg.jfb ? :jfb : :vanilla)
end

function construct(cfg::OptimizerConfig)
  if cfg.optimizer == "adam"
    opt = Optimisers.Adam(cfg.learning_rate)
  elseif cfg.opimizer == "sgd"
    if cfg.nesterov
      opt = Optimisers.Nesterov(cfg.learning_rate, cfg.momentum)
    elseif cfg.momentum == 0
      opt = Optimisers.Descent(cfg.learning_rate)
    else
      opt = Optimisers.Momentum(cfg.learning_rate, cfg.momentum)
    end
  else
    throw(ArgumentError("unknown value for `optimizer` = $(cfg.optimizer). Supported " *
                        "options are: `adam` and `sgd`."))
  end

  if cfg.weight_decay != 0
    opt = Optimisers.OptimiserChain(opt, Optimisers.WeightDecay(cfg.weight_decay))
  end

  if cfg.lr_scheduler == "cosine"
    scheduler = CosineAnnealSchedule(cfg.learning_rate, cfg.learning_rate / 100,
                                     cfg.cycle_length; dampen=1.2f0)
  elseif cfg.lr_scheduler == "constant"
    scheduler = ConstantSchedule(cfg.learning_rate)
  else
    throw(ArgumentError("unknown value for `lr_scheduler` = $(cfg.lr_scheduler). " *
                        "Supported options are: `constant` and `cosine`."))
  end

  return opt, scheduler
end

function get_loss_function(cfg::ExperimentConfig)
  if cfg.model.model_type == "vanilla"
    function _loss_function_vanilla(model, ps, st, (x, y))
      y_pred, st_ = model(x, ps, st)
      soln = st_.layer_2.solution
      ce_loss = logitcrossentropy(y_pred, y)
      skip_loss = mean_absolute_error(DEQ.initial_condition(soln),
                                      DEQ.equilibrium_solution(soln))
      loss = ce_loss
      nfe = DEQ.number_of_function_evaluations(soln)
      residual = DEQ.residual(soln)

      return (loss, st_, (; y_pred, nfe, ce_loss, skip_loss, residual))
    end

    return _loss_function_vanilla
  else
    function _loss_function_skip(model, ps, st, (x, y))
      y_pred, st_ = model(x, ps, st)
      soln = st_.layer_2.solution
      ce_loss = logitcrossentropy(y_pred, y)
      skip_loss = mean_absolute_error(DEQ.initial_condition(soln),
                                      DEQ.equilibrium_solution(soln))
      loss = ce_loss + cfg.train.w_skip * skip_loss
      nfe = DEQ.number_of_function_evaluations(soln)
      residual = DEQ.residual(soln)

      return (loss, st_, (; y_pred, nfe, ce_loss, skip_loss, residual))
    end

    return _loss_function_skip
  end
end
