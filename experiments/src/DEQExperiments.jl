module DEQExperiments

include("config.jl")
include("utils.jl")
include("construct.jl")
include("logging.jl")

# Patches
import CUDA, OneHotArrays

CUDA.unsafe_free!(x::OneHotArrays.OneHotArray) = CUDA.unsafe_free!(x.indices)

import ZygoteRules

# This is extremely unsafe and is needed rn to make WeightNorm work
ZygoteRules.gradtuple1(x::AbstractArray) = (x, nothing)

end
