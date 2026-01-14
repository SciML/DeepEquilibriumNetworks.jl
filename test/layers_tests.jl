@testsetup module LayersTestSetup

using NonlinearSolve, OrdinaryDiffEq

function loss_function(model, x, ps, st)
    y, st = model(x, ps, st)
    l1 = y isa Tuple ? sum(Base.Fix1(sum, abs2), y) : sum(abs2, y)
    l2 = st.solution.jacobian_loss
    l3 = sum(abs2, st.solution.z_star .- st.solution.u0)
    return l1 + l2 + l3
end

SOLVERS = (
    VCAB3(), Tsit5(), NewtonRaphson(; autodiff = AutoForwardDiff(; chunksize = 12)),
    SimpleLimitedMemoryBroyden(),
)

export loss_function, SOLVERS

end

@testitem "DEQ" setup = [SharedTestSetup, LayersTestSetup] begin
    using ADTypes, Lux, NonlinearSolve, OrdinaryDiffEq, SciMLSensitivity, Zygote

    rng = StableRNG(0)

    base_models = [
        Parallel(+, dense_layer(2 => 2), dense_layer(2 => 2)),
        Parallel(+, conv_layer((1, 1), 1 => 1), conv_layer((1, 1), 1 => 1)),
    ]
    init_models = [dense_layer(2 => 2), conv_layer((1, 1), 1 => 1)]
    x_sizes = [(2, 14), (3, 3, 1, 3)]

    model_type = (:deq, :skipdeq, :skipregdeq)
    _jacobian_regularizations = (nothing, AutoZygote(), AutoForwardDiff(), AutoFiniteDiff())

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        jacobian_regularizations = ongpu ? _jacobian_regularizations[1:(end - 1)] :
            _jacobian_regularizations

        @testset "Solver: $(nameof(typeof(solver))) | Model Type: $(mtype) | Jac. Reg: $(jacobian_regularization)" for solver in
                SOLVERS,
                mtype in model_type, jacobian_regularization in jacobian_regularizations

            @testset "x_size: $(x_size)" for (base_model, init_model, x_size) in
                zip(base_models, init_models, x_sizes)
                model = if mtype === :deq
                    DeepEquilibriumNetwork(base_model, solver; jacobian_regularization)
                elseif mtype === :skipdeq
                    SkipDeepEquilibriumNetwork(base_model, init_model, solver; jacobian_regularization)
                elseif mtype === :skipregdeq
                    SkipDeepEquilibriumNetwork(base_model, solver; jacobian_regularization)
                end

                ps, st = Lux.setup(rng, model) |> dev
                @test st.solution == DeepEquilibriumSolution()

                x = randn(rng, Float32, x_size...) |> dev
                z, st = model(x, ps, st)

                opt_broken = jacobian_regularization isa AutoZygote
                @jet model(x, ps, st) opt_broken = opt_broken

                @test all(isfinite, z)
                @test size(z) == size(x)
                @test st.solution isa DeepEquilibriumSolution
                @test maximum(abs, st.solution.residual) ≤ 1.0e-3

                _, gs_x, gs_ps, _ = Zygote.gradient(loss_function, model, x, ps, st)

                @test is_finite_gradient(gs_x)
                @test is_finite_gradient(gs_ps)

                ps, st = Lux.setup(rng, model) |> dev
                st = Lux.update_state(st, :fixed_depth, Val(10))
                @test st.solution == DeepEquilibriumSolution()

                z, st = model(x, ps, st)
                opt_broken = jacobian_regularization isa AutoZygote
                @jet model(x, ps, st) opt_broken = opt_broken

                @test all(isfinite, z)
                @test size(z) == size(x)
                @test st.solution isa DeepEquilibriumSolution
                @test st.solution.nfe == 10

                _, gs_x, gs_ps, _ = Zygote.gradient(loss_function, model, x, ps, st)

                @test is_finite_gradient(gs_x)
                @test is_finite_gradient(gs_ps)
            end
        end
    end
end

@testitem "Multiscale DEQ" setup = [SharedTestSetup, LayersTestSetup] begin
    using ADTypes, Lux, NonlinearSolve, OrdinaryDiffEq, SciMLSensitivity, Zygote

    rng = StableRNG(0)

    main_layers = [
        (
            Parallel(+, dense_layer(4 => 4), dense_layer(4 => 4)),
            dense_layer(3 => 3), dense_layer(2 => 2), dense_layer(1 => 1),
        ),
    ]

    mapping_layers = [
        [
            NoOpLayer() dense_layer(4 => 3) dense_layer(4 => 2) dense_layer(4 => 1);
            dense_layer(3 => 4) NoOpLayer() dense_layer(3 => 2) dense_layer(3 => 1);
            dense_layer(2 => 4) dense_layer(2 => 3) NoOpLayer() dense_layer(2 => 1);
            dense_layer(1 => 4) dense_layer(1 => 3) dense_layer(1 => 2) NoOpLayer()
        ],
    ]

    init_layers = [
        (
            dense_layer(4 => 4), dense_layer(4 => 3), dense_layer(4 => 2), dense_layer(4 => 1),
        ),
    ]

    x_sizes = [(4, 3)]
    scales = [((4,), (3,), (2,), (1,))]

    model_type = (:deq, :skipdeq, :skipregdeq, :node)
    jacobian_regularizations = (nothing,)

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        @testset "Solver: $(nameof(typeof(solver)))" for solver in SOLVERS,
                mtype in model_type, jacobian_regularization in jacobian_regularizations

            @testset "x_size: $(x_size)" for (
                    main_layer, mapping_layer, init_layer, x_size, scale,
                ) in zip(
                    main_layers, mapping_layers, init_layers, x_sizes, scales
                )
                model = if mtype === :deq
                    MultiScaleDeepEquilibriumNetwork(
                        main_layer, mapping_layer, nothing,
                        solver, scale; jacobian_regularization
                    )
                elseif mtype === :skipdeq
                    MultiScaleSkipDeepEquilibriumNetwork(
                        main_layer, mapping_layer, nothing, init_layer,
                        solver, scale; jacobian_regularization
                    )
                elseif mtype === :skipregdeq
                    MultiScaleSkipDeepEquilibriumNetwork(
                        main_layer, mapping_layer, nothing,
                        solver, scale; jacobian_regularization
                    )
                elseif mtype === :node
                    solver isa SciMLBase.AbstractODEAlgorithm || continue
                    MultiScaleNeuralODE(
                        main_layer, mapping_layer, nothing,
                        solver, scale; jacobian_regularization
                    )
                end

                ps, st = Lux.setup(rng, model) |> dev
                @test st.solution == DeepEquilibriumSolution()

                x = randn(rng, Float32, x_size...) |> dev
                z, st = model(x, ps, st)
                z_ = DEQs.flatten_vcat(z)

                @jet model(x, ps, st)

                @test all(isfinite, z_)
                @test size(z_) == (sum(prod, scale), size(x, ndims(x)))
                @test st.solution isa DeepEquilibriumSolution
                if st.solution.residual !== nothing
                    @test maximum(abs, st.solution.residual) ≤ 1.0e-3
                end

                _, gs_x, gs_ps, _ = Zygote.gradient(loss_function, model, x, ps, st)

                @test is_finite_gradient(gs_x)
                @test is_finite_gradient(gs_ps)

                ps, st = Lux.setup(rng, model) |> dev
                st = Lux.update_state(st, :fixed_depth, Val(10))
                @test st.solution == DeepEquilibriumSolution()

                z, st = model(x, ps, st)
                z_ = DEQs.flatten_vcat(z)
                @jet model(x, ps, st)

                @test all(isfinite, z_)
                @test size(z_) == (sum(prod, scale), size(x, ndims(x)))
                @test st.solution isa DeepEquilibriumSolution
                @test st.solution.nfe == 10

                _, gs_x, gs_ps, _ = Zygote.gradient(loss_function, model, x, ps, st)

                @test is_finite_gradient(gs_x)
                @test is_finite_gradient(gs_ps)
            end
        end
    end
end
