module DEQExperiments

include("config.jl")
include("utils.jl")
include("construct.jl")
include("logging.jl")

# Patches
import CUDA, OneHotArrays

CUDA.unsafe_free!(x::OneHotArrays.OneHotArray) = CUDA.unsafe_free!(x.indices)

end
