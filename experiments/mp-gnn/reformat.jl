using Glob, PyCall, Serialization

@pyimport pickle

function dump_pickle(filename, obj)
    out = open(filename,"w")
    pickle.dump(obj, out)
    close(out)
 end

function unpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end

for x in [3402, 27430, 46744]
    for f in glob("data/mp/$x/*.pkl")
        nf = replace(f, ".pkl" => ".jls")
        obj = unpickle(f)
        open(nf, "w") do io
            serialize(io, obj)
        end
        println("$f --> $nf")
    end
end
