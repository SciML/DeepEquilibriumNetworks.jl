module FastDEQ

using ChemistryFeaturization
using CUDA
using DataDeps
using DiffEqBase
using DiffEqSensitivity
using Flux
using FluxMPI
using Format
using Functors
using LinearAlgebra
using LinearSolve
using MPI
using OrdinaryDiffEq
using SciMLBase
using SparseArrays
using Statistics
using SteadyStateDiffEq
using UnPack
using Zygote

using Flux: hasaffine, ones32, zeros32, _isactive

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
    return Zygote.hook(Δ -> (_update_is_in_deq(!val); return Δ),
                       _update_is_in_deq(val))
end

@inline function _update_is_mask_reset_allowed(val::Bool)
    return allow_mask_reset[] = val
end

@inline function update_is_mask_reset_allowed(val::Bool)
    return Zygote.hook(Δ -> (_update_is_mask_reset_allowed(!val); return Δ),
                       _update_is_mask_reset_allowed(val))
end

@inline function _update_is_weightnorm_update_allowed(val::Bool)
    return allow_weightnorm_updates[] = val
end

@inline function update_is_weightnorm_update_allowed(val::Bool)
    return Zygote.hook(Δ -> (_update_is_weightnorm_update_allowed(!val); return Δ),
                       _update_is_weightnorm_update_allowed(val))
end

abstract type AbstractDeepEquilibriumNetwork end

function destructure end

function Base.show(io::IO, l::AbstractDeepEquilibriumNetwork)
    p, _ = destructure(l)
    return print(io, string(typeof(l).name.name), "() ", string(length(p)),
                 " Trainable Parameters")
end

Flux.trainable(d::AbstractDeepEquilibriumNetwork) = (d.p,)

abstract type IterativeDEQSolver end

include("utils.jl")
include("destructure.jl")

include("solvers/broyden.jl")
include("solvers/limited_memory_broyden.jl")

include("models/basics.jl")

include("layers/agn.jl")
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
include("models/cgcnn.jl")

include("losses.jl")
include("logger.jl")

export DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork
export MultiScaleDeepEquilibriumNetwork
export MultiScaleSkipDeepEquilibriumNetwork
export AGNConv, AGNMaxPool, AGNMeanPool
export VariationalHiddenDropout, GroupNormV2, WeightNorm

export DEQChain
export BasicResidualBlock, BranchNet, MultiParallelNet
export downsample_module, upsample_module, expand_channels_module, conv3x3,
       conv5x5
export WidthStackedDEQ
export CrystalGraphCNN

export batch_graph_data, BatchedAtomicGraph
export reset_mask!, get_and_clear_nfe!, get_default_ssadjoint,
       get_default_dynamicss_solver, get_default_ssrootfind_solver

export PrettyTableLogger

export SupervisedLossContainer

export BroydenSolver, BroydenCache
export LimitedMemoryBroydenSolver, LimitedMemoryBroydenCache

end
