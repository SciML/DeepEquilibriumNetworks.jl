function _should_log(;logging_rank = 0, comm = MPI.COMM_WORLD)
    MPI.Initialized() || return true # Not using MPI
    return MPI.Comm_rank(comm) == logging_rank
end

# Running AverageMeter
mutable struct AverageMeter{T}
    last_value::T
    sum::T
    count::Int

    AverageMeter(T = Float32) = new{T}(T(0), T(0), 0)
end

function reset!(am::AverageMeter{T}) where {T}
    am.last_value = T(0)
    am.sum = T(0)
    am.count = 0
    return am
end

function update!(am::AverageMeter{T}, val::T) where {T}
    am.last_value = val
    am.sum += val
    am.count += 1
    return am.sum / am.count
end

(am::AverageMeter)() = am.sum / am.count


# Simple Table Logger
## TODO: Writing to a file
struct PrettyTableLogger{N,AM,F,R}
    header::NTuple{N,String}
    average_meters::AM
    span::Int
    fmtrfuncs::F
    records::R

    function PrettyTableLogger(header, record = [])
        N = length(header) + length(record)
        ind_lens = vcat(length.(header), length.(record))
        span = sum(ind_lens .+ 3) + 1
        println("="^span)
        headers = vcat(header, record)
        for h in headers
            print("| $h ")
        end
        println("|")
        println("="^span)

        avg_meters =
            Dict{String,AverageMeter}(rec => AverageMeter() for rec in record)

        patterns = ["%$l.4f" for l in ind_lens]
        fmtrfuncs = generate_formatter.(patterns)

        record = tuple(record...)

        return new{N,typeof(avg_meters),typeof(fmtrfuncs),typeof(record)}(
            tuple(headers...),
            avg_meters,
            span,
            fmtrfuncs,
            record
        )
    end
end

function (pl::PrettyTableLogger)(args...; last::Bool = false, records::Dict = Dict())
    _should_log() || return
    if length(records) > 0
        for (rec, val) in records
            update!(pl.average_meters[rec], val)
        end
        return
    end
    if last
        println("="^pl.span)
        return
    end
    for h in [
        fmtrfunc(arg) for (fmtrfunc, arg) in
        zip(pl.fmtrfuncs, vcat([args...], [pl.average_meters[rec]() for rec in pl.records]))
    ]
        print("| $h ")
    end
    println("|")
end
