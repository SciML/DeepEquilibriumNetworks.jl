import Dates, Formatting, Logging, LoggingExtras, Wandb

# Average Meter
Base.@kwdef mutable struct AverageMeter
  fmtstr::Any
  val::Float64 = 0.0
  sum::Float64 = 0.0
  count::Int = 0
  average::Float64 = 0
end

function AverageMeter(name::String, fmt::String)
  fmtstr = Formatting.FormatExpr("$name {1:$fmt} ({2:$fmt})")
  return AverageMeter(; fmtstr=fmtstr)
end

function (meter::AverageMeter)(val, n::Int)
  meter.val = val
  meter.sum += val * n
  meter.count += n
  meter.average = meter.sum / meter.count
  return meter.average
end

function reset_meter!(meter::AverageMeter)
  meter.val = 0.0
  meter.sum = 0.0
  meter.count = 0
  meter.average = 0.0
  return meter
end

function print_meter(meter::AverageMeter)
  return Formatting.printfmt(meter.fmtstr, meter.val, meter.average)
end

# ProgressMeter
struct ProgressMeter{N}
  batch_fmtstr::Any
  meters::NTuple{N, AverageMeter}
end

function ProgressMeter(num_batches::Int, meters::NTuple{N}, prefix::String="") where {N}
  fmt = "%" * string(length(string(num_batches))) * "d"
  prefix = prefix != "" ? endswith(prefix, " ") ? prefix : prefix * " " : ""
  batch_fmtstr = Formatting.generate_formatter("$prefix[$fmt/" *
                                               Formatting.sprintf1(fmt, num_batches) *
                                               "]")
  return ProgressMeter{N}(batch_fmtstr, meters)
end

function reset_meter!(meter::ProgressMeter)
  reset_meter!.(meter.meters)
  return meter
end

function print_meter(meter::ProgressMeter, batch::Int)
  base_str = meter.batch_fmtstr(batch)
  print(base_str)
  foreach(x -> (print("\t"); print_meter(x)), meter.meters[1:end])
  println()
  return nothing
end

get_loggable_values(meter::ProgressMeter) = getproperty.(meter.meters, :average)

# Log to a CSV File
struct CSVLogger{N}
  filename::Any
  fio::Any
end

function CSVLogger(filename, header)
  should_log() && !isdir(dirname(filename)) && mkpath(dirname(filename))
  fio = should_log() ? open(filename, "w") : nothing
  N = length(header)
  should_log() && println(fio, join(header, ","))
  return CSVLogger{N}(filename, fio)
end

function (csv::CSVLogger)(args...)
  if should_log()
    println(csv.fio, join(args, ","))
    flush(csv.fio)
  end
end

function Base.close(csv::CSVLogger)
  if should_log()
    close(csv.fio)
  end
end

function create_logger(base_dir::String, train_length::Int, eval_length::Int,
                       expt_name::String, config::Dict)
  if !isdir(base_dir)
    @warn "$(base_dir) doesn't exist. Creating a directory."
    mkpath(base_dir)
  end

  @info expt_name
  @info config

  # Wandb Logger
  wandb_logger = Wandb.WandbLogger(; project="skipdeq", name=expt_name, config=config)

  # CSV Logger
  train_csv_header = [
    "Step",
    "Batch Time",
    "Data Time",
    "Forward Pass Time",
    "Backward Pass Time",
    "Optimizer Time",
    "Cross Entropy Loss",
    "Skip Loss",
    "Net Loss",
    "NFE",
    "Accuracy (Top 1)",
    "Accuracy (Top 5)",
    "Residual",
  ]
  train_loggable_dict(args...) = Dict(zip(.*(("Train/",), train_csv_header), args))
  csv_logger_train = CSVLogger(joinpath(base_dir, "results_train.csv"), train_csv_header)

  eval_csv_header = [
    "Step",
    "Batch Time",
    "Data Time",
    "Forward Pass Time",
    "Cross Entropy Loss",
    "Skip Loss",
    "Net Loss",
    "NFE",
    "Accuracy (Top 1)",
    "Accuracy (Top 5)",
    "Residual",
  ]
  eval_loggable_dict(args...) = Dict(zip(.*(("Eval/",), eval_csv_header), args))
  csv_logger_eval = CSVLogger(joinpath(base_dir, "results_eval.csv"), eval_csv_header)

  # Train Logger
  batch_time = AverageMeter("Batch Time", "6.3f")
  data_time = AverageMeter("Data Time", "6.3f")
  fwd_time = AverageMeter("Forward Pass Time", "6.3f")
  bwd_time = AverageMeter("Backward Pass Time", "6.3f")
  opt_time = AverageMeter("Optimizer Time", "6.3f")
  loss = AverageMeter("Net Loss", "6.3f")
  ce_loss = AverageMeter("Cross Entropy Loss", "6.3e")
  skip_loss = AverageMeter("Skip Loss", "6.3e")
  residual = AverageMeter("Residual", "6.3e")
  top1 = AverageMeter("Accuracy (@1)", "3.2f")
  top5 = AverageMeter("Accuracy (@5)", "3.2f")
  nfe = AverageMeter("NFE", "3.2f")

  progress = ProgressMeter(train_length,
                           (batch_time, data_time, fwd_time, bwd_time, opt_time, ce_loss,
                            skip_loss, loss, nfe, top1, top5, residual), "Train:")

  train_logger = (progress=progress,
                  avg_meters=(; batch_time, data_time, loss, ce_loss, skip_loss, residual,
                              top1, top5, nfe, fwd_time, bwd_time, opt_time))

  # Eval Logger
  batch_time = AverageMeter("Batch Time", "6.3f")
  data_time = AverageMeter("Data Time", "6.3f")
  fwd_time = AverageMeter("Forward Time", "6.3f")
  loss = AverageMeter("Net Loss", "6.3f")
  ce_loss = AverageMeter("Cross Entropy Loss", "6.3e")
  skip_loss = AverageMeter("Skip Loss", "6.3e")
  residual = AverageMeter("Residual", "6.3e")
  top1 = AverageMeter("Accuracy (@1)", "3.2f")
  top5 = AverageMeter("Accuracy (@5)", "3.2f")
  nfe = AverageMeter("NFE", "3.2f")

  progress = ProgressMeter(eval_length,
                           (batch_time, data_time, fwd_time, ce_loss, skip_loss, loss, nfe,
                            top1, top5, residual), "Test:")

  eval_logger = (progress=progress,
                 avg_meters=(; batch_time, data_time, fwd_time, loss, ce_loss, skip_loss,
                             residual, top1, top5, nfe))

  return (csv_loggers=(; train=csv_logger_train, eval=csv_logger_eval),
          wandb_logger=wandb_logger,
          progress_loggers=(; train=train_logger, eval=eval_logger),
          log_functions=(; train=train_loggable_dict, eval=eval_loggable_dict))
end
