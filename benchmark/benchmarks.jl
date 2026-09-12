using DeepEquilibriumNetworks, BenchmarkTools
using Lux, Random, StableRNGs
using NonlinearSolve, OrdinaryDiffEqTsit5

const SUITE = BenchmarkGroup()
const rng = StableRNG(0)

function dense_layer(args...; kwargs...)
    init_weight(rng::AbstractRNG, dims...) = randn(rng, Float32, dims) .* 0.001f0
    return Dense(args...; init_weight, use_bias = false, kwargs...)
end

# =============================================================================
# DeepEquilibriumNetwork — construction, setup, forward (fixed-point solve)
# =============================================================================

model = Parallel(+, dense_layer(2 => 2), dense_layer(2 => 2))
deq = DeepEquilibriumNetwork(model, Tsit5())
skip_deq = SkipDeepEquilibriumNetwork(model, dense_layer(2 => 2), Tsit5())
nlsolve_deq = DeepEquilibriumNetwork(
    Parallel(+, dense_layer(4 => 4), dense_layer(4 => 4)),
    NewtonRaphson()
)

ps, st = Lux.setup(rng, deq)
ps_s, st_s = Lux.setup(rng, skip_deq)
ps_n, st_n = Lux.setup(rng, nlsolve_deq)

x = randn(rng, Float32, 2, 4)
x_n = randn(rng, Float32, 4, 4)

SUITE["deq"] = BenchmarkGroup()

SUITE["deq"]["construct"] = @benchmarkable DeepEquilibriumNetwork(
    $model, Tsit5()
)
SUITE["deq"]["construct_skip"] = @benchmarkable SkipDeepEquilibriumNetwork(
    $model, dense_layer(2 => 2), Tsit5()
)
SUITE["deq"]["setup"] = @benchmarkable Lux.setup($rng, $deq)
SUITE["deq"]["forward"] = @benchmarkable $deq($x, $ps, $st)
SUITE["deq"]["forward_skip"] = @benchmarkable $skip_deq($x, $ps_s, $st_s)
SUITE["deq"]["forward_newton"] = @benchmarkable $nlsolve_deq($x_n, $ps_n, $st_n)
