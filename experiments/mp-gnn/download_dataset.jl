# Materials Project API key should be stored in an environment variable `MP_API_KEY`
using HTTP, JSON

load_ids(filepath::String) = readlines(open(filepath))

function make_api_call(id::String)
    API_KEY = ENV["MP_API_KEY"]
    try
        return HTTP.get("https://materialsproject.org/rest/v2/materials/$id/vasp?API_KEY=$API_KEY")
    catch e
        return ""
    end
end

download_dataset(filepath::String) = download_dataset(load_ids(filepath))

function download_dataset(ids::Vector{String})
    responses = Vector{Any}(undef, length(ids))
    Threads.@threads for i in 1:length(ids)
        i % 100 == 0 && println("Fetching Response: $i")
        responses[i] = make_api_call(ids[i])
    end
    return responses
end

function parse_response(response, basedir, target_property)
    r = JSON.parse(String(copy(response.body)))["response"]
    length(r) == 0 && return (-1, -1)
    r = r[1]
    f = open(joinpath(basedir, r["material_id"] * ".cif"), "w")
    write(f, r["cif"])
    close(f)
    return r["material_id"], r[target_property]
end

function parse_response(response::Vector, basedir::String, target_property::String)
    target_property_map = Dict{String, Any}()
    Threads.@threads for i in 1:length(response)
        response[i] == "" && continue
        id, prop = parse_response(response[i], basedir, target_property)
        id != -1 && (target_property_map[id] = prop)
    end
    open(joinpath(basedir, "id_prop.csv"), "w") do f
        for (k, v) in target_property_map
            println(f, "$k, $v")
        end
    end
end

for count in [3402, 27430, 46744]
    filepath = "experiments/mp-gnn/mp-ids-$count.txt"
    dataset = download_dataset(filepath)
    parse_response(dataset, "data/mp/$count/", "formation_energy_per_atom")
end
