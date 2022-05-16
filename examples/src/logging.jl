Base.@kwdef mutable struct AverageMeter
    fmtstr
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

print_meter(meter::AverageMeter) = Formatting.printfmt(meter.fmtstr, meter.val, meter.average)

struct ProgressMeter{N}
    batch_fmtstr
    meters::NTuple{N,AverageMeter}
end

function ProgressMeter(num_batches::Int, meters::NTuple{N}, prefix::String="") where {N}
    fmt = "%" * string(length(string(num_batches))) * "d"
    prefix = prefix != "" ? endswith(prefix, " ") ? prefix : prefix * " " : ""
    batch_fmtstr = Formatting.generate_formatter("$prefix[$fmt/" * sprintf1(fmt, num_batches) * "]")
    return ProgressMeter{N}(batch_fmtstr, meters)
end

function print_meter(meter::ProgressMeter, batch::Int)
    base_str = meter.batch_fmtstr(batch)
    print(base_str)
    foreach(x -> (print("\t"); print_meter(x)), meter.meters[1:end])
    return println()
end

struct CSVLogger{N}
    filename
    fio
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
