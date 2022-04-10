
function _should_log(; logging_rank=0, comm=MPI.COMM_WORLD)
    FluxMPI.Initialized() || return true # Not using MPI
    return local_rank() == logging_rank
end

# Running AverageMeter
mutable struct AverageMeter{T}
    last_value::T
    sum::T
    count::Int

    AverageMeter(T=Float32) = new{T}(T(0), T(0), 0)
end

function reset!(am::AverageMeter{T}) where {T}
    val = am()
    am.last_value = T(0)
    am.sum = T(0)
    am.count = 0
    return val
end

function update!(am::AverageMeter{T}, val::T) where {T}
    am.last_value = val
    am.sum += val
    am.count += 1
    return am.sum / am.count
end

update!(am::AverageMeter{T}, val) where {T} = update!(am, T(val))

(am::AverageMeter)() = am.sum / am.count

# Simple Table Logger
struct PrettyTableLogger{N,AM,F,R,FIO}
    header::NTuple{N,String}
    average_meters::AM
    span::Int
    fmtrfuncs::F
    records::R
    fio::FIO

    function PrettyTableLogger(filename::String, header, record=[])
        fio = _should_log() ? open(filename, "w") : nothing

        N = length(header) + length(record)
        headers = vcat(header, record)
        headers_og = headers
        _c = 0
        count() = (_c += 1; _c)
        rsplits = first.(map(x -> length(x) >= 2 ? x : ("__" * string(count()), x), rsplit.(headers, "/"; limit=2)),)
        headers = string.(last.(rsplit.(headers, "/"; limit=2)))
        headers = map(x -> length(x) <= 6 ? x * (" "^length(x)) : x, headers)
        ind_lens = length.(headers)
        span = sum(ind_lens .+ 3) + 1
        rsplit_lens = Dict()
        if fio !== nothing
            for (i, r) in enumerate(rsplits)
                _r = string(r)
                _r ∉ keys(rsplit_lens) && (rsplit_lens[_r] = -3 - length(_r) + 1)
                rsplit_lens[_r] = rsplit_lens[_r] + ind_lens[i] + 3
            end
            rsplits_unique = unique(rsplits)
            if !(length(rsplits_unique) == 1 && rsplits_unique[0] == "")
                println("="^span)
                for r in rsplits_unique
                    if startswith(r, "__")
                        print("| " * (" "^length(r)) * (" "^rsplit_lens[string(r)]))
                    else
                        print("| $r" * (" "^rsplit_lens[string(r)]))
                    end
                end
                println("|")
            end
            println("="^span)
            for h in headers
                print("| $h ")
            end
            println("|")
            println("="^span)
            for h in headers_og[1:(end - 1)]
                print(fio, "$h,")
            end
            println(fio, "$(headers_og[end])")
        end

        avg_meters = Dict{String,AverageMeter}(rec => AverageMeter() for rec in record)

        patterns = ["%$l.4f" for l in ind_lens]
        fmtrfuncs = generate_formatter.(patterns)

        record = tuple(record...)

        return new{N,typeof(avg_meters),typeof(fmtrfuncs),typeof(record),typeof(fio)}(
            tuple(headers...), avg_meters, span, fmtrfuncs, record, fio
        )
    end
end

function (pl::PrettyTableLogger)(args...; last::Bool=false, records::Dict=Dict())
    _should_log() || return nothing
    if length(records) > 0
        for (rec, val) in records
            update!(pl.average_meters[rec], val)
        end
        return nothing
    end
    if last
        str = "="^pl.span
        println(str)
        return nothing
    end
    for (i, (fmtrfunc, arg)) in
        enumerate(zip(pl.fmtrfuncs, vcat([args...], [reset!(pl.average_meters[rec]) for rec in pl.records])))
        h = fmtrfunc(arg)
        print("| $h ")
        if i < length(pl.fmtrfuncs)
            print(pl.fio, "$arg,")
        else
            println(pl.fio, "$arg")
        end
    end
    println("|")
    flush(pl.fio)
    return nothing
end

function Base.close(pl::PrettyTableLogger)
    pl(; last=true)
    pl.fio === nothing || close(pl.fio)
    return nothing
end

function Base.show(io, pl::PrettyTableLogger)
    print(io, "PrettyTableLogger(", pl.fio, ")")
end
