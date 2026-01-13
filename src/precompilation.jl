using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    @compile_workload begin
        # Precompile core functionality for DeepEquilibriumNetwork
        # These are the most common operations users perform

        # Simple Dense-based DEQ model setup
        rng = Random.Xoshiro(0)

        # Create a small model for precompilation
        # Using SSRootfind which is already imported from SteadyStateDiffEq
        model = DEQ(
            Parallel(+, Lux.Dense(2, 2; use_bias = false), Lux.Dense(2, 2; use_bias = false)),
            SSRootfind();
            verbose = false
        )

        # Initialize parameters and state (very common operation)
        ps, st = LuxCore.setup(rng, model)

        # Precompile DeepEquilibriumSolution constructor
        _ = DeepEquilibriumSolution()

        # Precompile utility functions
        x = ones(Float32, 2, 1)

        # Precompile check_unrolled_mode
        _ = check_unrolled_mode(st)

        # Precompile zeros_init
        _ = zeros_init(nothing, x)

        # Precompile flatten operations
        _ = flatten(x)
        _ = flatten_vcat((x, x))

        # Precompile split_and_reshape with Nothing
        _ = split_and_reshape(x, nothing, nothing)

        # Precompile with fixed depth (unrolled mode)
        st_unrolled = Lux.update_state(st, :fixed_depth, Val(2))
        _ = check_unrolled_mode(st_unrolled)
    end
end
