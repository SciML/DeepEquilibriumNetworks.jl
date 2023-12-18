using ADTypes, DeepEquilibriumNetworks, DiffEqBase, NonlinearSolve, OrdinaryDiffEq,
    SciMLBase, Test, NLsolve

include("test_utils.jl")

@testset "DeepEquilibriumNetwork" begin
    rng = __get_prng(0)

    base_models = [
        Parallel(+, __get_dense_layer(2 => 2), __get_dense_layer(2 => 2)),
        Parallel(+, __get_conv_layer((1, 1), 1 => 1), __get_conv_layer((1, 1), 1 => 1)),
    ]
    init_models = [__get_dense_layer(2 => 2), __get_conv_layer((1, 1), 1 => 1)]
    x_sizes = [(2, 14), (3, 3, 1, 3)]

    model_type = (:deq, :skipdeq, :skipregdeq)
    solvers = (VCAB3(), Tsit5(), NewtonRaphson(), NLsolveJL(), SimpleLimitedMemoryBroyden())
    jacobian_regularizations = (nothing, AutoFiniteDiff(), AutoZygote())

    @testset "Solver: $(__nameof(solver))" for solver in solvers,
        mtype in model_type, jacobian_regularization in jacobian_regularizations

        @testset "x_size: $(x_size)" for (base_model, init_model, x_size) in zip(base_models,
            init_models, x_sizes)
            model = if mtype === :deq
                DeepEquilibriumNetwork(base_model, solver; jacobian_regularization)
            elseif mtype === :skipdeq
                SkipDeepEquilibriumNetwork(base_model, init_model, solver;
                    jacobian_regularization)
            elseif mtype === :skipregdeq
                SkipDeepEquilibriumNetwork(base_model, solver; jacobian_regularization)
            end

            ps, st = Lux.setup(rng, model)
            @test st.solution == DeepEquilibriumSolution()

            x = randn(rng, Float32, x_size...)
            z, st = model(x, ps, st)

            opt_broken = solver isa NewtonRaphson ||
                         solver isa SimpleLimitedMemoryBroyden ||
                         jacobian_regularization isa AutoZygote
            @jet model(x, ps, st) opt_broken=opt_broken # Broken due to nfe dynamic dispatch

            @test all(isfinite, z)
            @test size(z) == size(x)
            @test st.solution isa DeepEquilibriumSolution
            @test maximum(abs, st.solution.residual) ≤ 1e-3

            ps, st = Lux.setup(rng, model)
            st = Lux.update_state(st, :fixed_depth, Val(10))
            @test st.solution == DeepEquilibriumSolution()

            z, st = model(x, ps, st)
            @jet model(x, ps, st)

            @test all(isfinite, z)
            @test size(z) == size(x)
            @test st.solution isa DeepEquilibriumSolution
            @test st.solution.nfe == 10
        end
    end
end

@testset "MultiScaleDeepEquilibriumNetworks" begin
    rng = __get_prng(0)

    main_layers = [
        (Parallel(+, __get_dense_layer(4 => 4), __get_dense_layer(4 => 4)),
            __get_dense_layer(3 => 3), __get_dense_layer(2 => 2),
            __get_dense_layer(1 => 1)),
    ]

    mapping_layers = [
        [NoOpLayer() __get_dense_layer(4 => 3) __get_dense_layer(4 => 2) __get_dense_layer(4 => 1);
            __get_dense_layer(3 => 4) NoOpLayer() __get_dense_layer(3 => 2) __get_dense_layer(3 => 1);
            __get_dense_layer(2 => 4) __get_dense_layer(2 => 3) NoOpLayer() __get_dense_layer(2 => 1);
            __get_dense_layer(1 => 4) __get_dense_layer(1 => 3) __get_dense_layer(1 => 2) NoOpLayer()],
    ]

    init_layers = [
        (__get_dense_layer(4 => 4), __get_dense_layer(4 => 3), __get_dense_layer(4 => 2),
            __get_dense_layer(4 => 1)),
    ]

    x_sizes = [(4, 3)]
    scales = [((4,), (3,), (2,), (1,))]

    model_type = (:deq, :skipdeq, :skipregdeq, :node)
    solvers = (VCAB3(), Tsit5(), NewtonRaphson(), NLsolveJL(), SimpleLimitedMemoryBroyden())
    jacobian_regularizations = (nothing, AutoFiniteDiff(), AutoZygote())

    for mtype in model_type, jacobian_regularization in jacobian_regularizations

        @testset "Solver: $(__nameof(solver))" for solver in solvers

            @testset "x_size: $(x_size)" for (main_layer, mapping_layer, init_layer, x_size, scale) in zip(main_layers,
                mapping_layers, init_layers, x_sizes, scales)
                model = if mtype === :deq
                    MultiScaleDeepEquilibriumNetwork(main_layer, mapping_layer, nothing,
                        solver,
                        scale; jacobian_regularization)
                elseif mtype === :skipdeq
                    MultiScaleSkipDeepEquilibriumNetwork(main_layer, mapping_layer, nothing,
                        init_layer, solver, scale; jacobian_regularization)
                elseif mtype === :skipregdeq
                    MultiScaleSkipDeepEquilibriumNetwork(main_layer, mapping_layer, nothing,
                        solver, scale; jacobian_regularization)
                elseif mtype === :node
                    solver isa SciMLBase.AbstractODEAlgorithm || continue
                    MultiScaleNeuralODE(main_layer, mapping_layer, nothing, solver, scale;
                        jacobian_regularization)
                end

                ps, st = Lux.setup(rng, model)
                @test st.solution == DeepEquilibriumSolution()

                x = randn(rng, Float32, x_size...)
                z, st = model(x, ps, st)
                z_ = mapreduce(DEQs.__flatten, vcat, z)

                opt_broken = solver isa NewtonRaphson ||
                             solver isa SimpleLimitedMemoryBroyden ||
                             jacobian_regularization isa AutoZygote
                @jet model(x, ps, st) opt_broken=opt_broken # Broken due to nfe dynamic dispatch

                @test all(isfinite, z_)
                @test size(z_) == (sum(prod, scale), size(x, ndims(x)))
                @test st.solution isa DeepEquilibriumSolution
                if st.solution.residual !== nothing
                    @test maximum(abs, st.solution.residual) ≤ 1e-3
                end

                ps, st = Lux.setup(rng, model)
                st = Lux.update_state(st, :fixed_depth, Val(10))
                @test st.solution == DeepEquilibriumSolution()

                z, st = model(x, ps, st)
                z_ = mapreduce(DEQs.__flatten, vcat, z)
                opt_broken = jacobian_regularization isa AutoZygote
                @jet model(x, ps, st) opt_broken=opt_broken

                @test all(isfinite, z_)
                @test size(z_) == (sum(prod, scale), size(x, ndims(x)))
                @test st.solution isa DeepEquilibriumSolution
                @test st.solution.nfe == 10
            end
        end
    end
end
