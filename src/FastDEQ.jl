module FastDEQ

using CUDA
using DiffEqBase
using DiffEqSensitivity
using Flux
using FluxExperimental
using FluxMPI
using Format
using Functors
using LinearAlgebra
using LinearSolve
using MPI
using OrdinaryDiffEq
using Requires
using SciMLBase
using SparseArrays
using Statistics
using SteadyStateDiffEq
using UnPack
using Zygote

using Flux: hasaffine, ones32, zeros32, _isactive

function __init__()
    @require ChemistryFeaturization="6c925690-434a-421d-aea7-51398c5b007a" begin
        include("layers/agn.jl")
        include("models/cgcnn.jl")
        # Atomic Graph Net Layers
        export AGNConv, AGNMaxPool, AGNMeanPool, batch_graph_data, BatchedAtomicGraph
        export CrystalGraphCNN
    end
end

const is_inside_deq = Ref(false)
const allow_mask_reset = Ref(false)
const allow_weightnorm_updates = Ref(false)

@inline is_in_deq() = is_inside_deq[]
@inline is_mask_reset_allowed() = allow_mask_reset[]
@inline is_weightnorm_update_allowed() = allow_weightnorm_updates[]

@inline function _update_is_in_deq(val::Bool)
    return is_inside_deq[] = val
end

@inline function update_is_in_deq(val::Bool)
    return Zygote.hook(Δ -> (_update_is_in_deq(!val); return Δ), _update_is_in_deq(val))
end

@inline function _update_is_mask_reset_allowed(val::Bool)
    return allow_mask_reset[] = val
end

@inline function update_is_mask_reset_allowed(val::Bool)
    return Zygote.hook(Δ -> (_update_is_mask_reset_allowed(!val); return Δ), _update_is_mask_reset_allowed(val))
end

@inline function _update_is_weightnorm_update_allowed(val::Bool)
    return allow_weightnorm_updates[] = val
end

@inline function update_is_weightnorm_update_allowed(val::Bool)
    return Zygote.hook(Δ -> (_update_is_weightnorm_update_allowed(!val); return Δ),
                       _update_is_weightnorm_update_allowed(val))
end

abstract type AbstractDeepEquilibriumNetwork end

function Base.show(io::IO, l::AbstractDeepEquilibriumNetwork)
    p, _ = destructure_parameters(l)
    return print(io, string(typeof(l).name.name), "() ", string(length(p)), " Trainable Parameters")
end

Flux.trainable(d::AbstractDeepEquilibriumNetwork) = (d.p,)

abstract type IterativeDEQSolver end

include("utils.jl")

include("solvers/broyden.jl")
include("solvers/limited_memory_broyden.jl")

include("models/basics.jl")

include("layers/jacobian_stabilization.jl")
include("layers/utils.jl")
include("layers/deq.jl")
include("layers/sdeq.jl")
include("layers/mdeq.jl")
include("layers/smdeq.jl")
include("layers/dropout.jl")
include("layers/normalise.jl")
include("layers/weight_norm.jl")

include("models/chain.jl")
include("models/width_stacked_deq.jl")

include("losses.jl")
include("logger.jl")

# DEQ Layers
export DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork, MultiScaleDeepEquilibriumNetwork,
       MultiScaleSkipDeepEquilibriumNetwork, WidthStackedDEQ
# General Purpose Layers (probably should be in Flux.jl)
export VariationalHiddenDropout, GroupNormV2, WeightNorm

# Compositional Layers
export DEQChain
export BasicResidualBlock, BranchNet, MultiParallelNet
export downsample_module, upsample_module, expand_channels_module, conv3x3, conv5x5

# For the sanity of experiment code
export reset_mask!, get_and_clear_nfe!, get_default_ssadjoint, get_default_dynamicss_solver,
       get_default_ssrootfind_solver

# Text Logging
export PrettyTableLogger

# Again experiment code sanity
export SupervisedLossContainer

# Non Linear Solvers (need to find time to move into a dedicated package with a proper API)
export BroydenSolver, BroydenCache, LimitedMemoryBroydenSolver, LimitedMemoryBroydenCache

end
