module FastDEQ

using CUDA, Flux, LinearAlgebra, Zygote


include("utils.jl")
include("solvers.jl")
include("layers.jl")


export AndersonAcceleration, fixedpointsolve, DEQFixedPointLayer


end
