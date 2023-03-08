# Multi-Scale Neural ODE with Input Injection -- Baseline for Continuous MDEQ
# NOTE(@avik-pal): This is mostly for baseline comparision against Continuous MDEQ. Users
#                  mostly should not be using this implementation.
# struct MultiScaleNeuralODE{N, Sc, M, A, S, K} <:
#        Lux.AbstractExplicitContainerLayer{(:model,)}
#   model::M
#   solver::A
#   sensealg::S
#   scales::Sc
#   kwargs::K
# end

# function Lux.initialstates(rng::Random.AbstractRNG, node::MultiScaleNeuralODE)
#   return (model=Lux.initialstates(rng, node.model),
#           split_idxs=Static.static(Tuple(vcat(0, cumsum(prod.(node.scales))...))),
#           fixed_depth=Val(0), initial_condition=zeros(Float32, 1, 1), solution=nothing)
# end

# function MultiScaleNeuralODE(main_layers::Tuple, mapping_layers::Matrix,
#                              post_fuse_layer::Union{Nothing, Tuple}, solver,
#                              scales::NTuple{N, NTuple{L, Int64}};
#                              sensealg=SciMLSensitivity.InterpolatingAdjoint(;
#                                                                             autojacvec=SciMLSensitivity.ZygoteVJP()),
#                              kwargs...) where {N, L}
#   l1 = Lux.Parallel(nothing, main_layers...)
#   l2 = Lux.BranchLayer(Lux.Parallel.(+,
#                                      map(x -> tuple(x...), eachrow(mapping_layers))...)...)
#   if post_fuse_layer === nothing
#     model = Lux.Chain(l1, l2)
#   else
#     model = Lux.Chain(l1, l2, Lux.Parallel(nothing, post_fuse_layer...))
#   end
#   scales = Static.static(scales)
#   return MultiScaleNeuralODE{N, typeof(scales), typeof(model), typeof(solver),
#                              typeof(sensealg), typeof(kwargs)}(model, solver, sensealg,
#                                                                scales, kwargs)
# end

# function (node::MultiScaleNeuralODE{N})(x::AbstractArray{T}, ps,
#                                         st::NamedTuple) where {N, T}
#   z, st = _get_initial_condition_mdeq(node.scales, x, st)

#   if _check_unrolled_mode(st)
#     z_star = split_and_reshape(z, st.split_idxs, node.scales)
#     z_star, st_ = _evaluate_unrolled_mdeq(node.model, z_star, x, ps, st.model,
#                                           st.fixed_depth)

#     z_star_flatten = vcat(MLUtils.flatten.(z_star)...)
#     solution = DeepEquilibriumSolution(z_star_flatten, z, 0.0f0, 0.0f0,
#                                        _get_unrolled_depth(st))
#     st__ = merge(st, (; model=st_, solution))

#     return z_star, st__
#   end

#   st_ = st.model

#   function dudt(u, p, t)
#     u_split = split_and_reshape(u, st.split_idxs, node.scales)
#     return node.model(((u_split[1], x), u_split[2:N]...), p, st_)
#   end

#   dudt_(u, p, t) = vcat(MLUtils.flatten.(first(dudt(u, p, t)))...)

#   prob = OrdinaryDiffEq.ODEProblem(OrdinaryDiffEq.ODEFunction{false}(dudt_), z,
#                                    (0.0f0, 1.0f0), ps)
#   sol = SciMLBase.solve(prob, node.solver; node.sensealg, node.kwargs...)
#   z_star, st_ = dudt(sol.u[end], ps, nothing)

#   # The solution is not actually the Equilibrium Solution. Just makes life easier to call
#   # it that.
#   solution = DeepEquilibriumSolution(vcat(MLUtils.flatten.(z_star)...), z, 0.0f0, 0.0f0,
#                                      sol.destats.nf + 1)
#   st__ = merge(st, (; model=st_, solution))

#   return z_star, st__
# end
