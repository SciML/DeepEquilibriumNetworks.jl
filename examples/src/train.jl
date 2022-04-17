function update_lr(st::ST, eta) where {ST}
    if hasfield(ST, :eta)
        @set! st.eta = eta
    end
    return st
end

update_lr(st::Optimisers.OptimiserChain, eta) = update_lr.(st.opts, eta)

function update_lr(st::Optimisers.Leaf, eta)
    @set! st.rule = update_lr(st.rule, eta)
end

update_lr(st_opt::NamedTuple, eta) = fmap(l -> update_lr(l, eta), st_opt)

function construct_optimiser(config::ExperimentConfiguration)
    opt = if config.optimiser == :ADAM
        Optimisers.ADAM(config.eta)
    elseif config.optimiser == :SGD
        if config.nesterov
            Optimisers.Nesterov(config.eta, config.momentum)
        else
            if iszero(config.momentum)
                Optimisers.Descent(config.eta)
            else
                Optimisers.Momentum(config.eta, config.momentum)
            end
        end
    else
        throw(ArgumentError("`config.optimiser` must be either `:ADAM` or `:SGD`"))
    end
    if !iszero(config.weight_decay)
        opt = Optimisers.OptimiserChain(opt, Optimisers.WeightDecay(config.weight_decay))
    end

    sched = if config.lr_scheduler == :COSINE
        ParameterSchedulers.Stateful(ParameterSchedulers.Cos(config.eta, 1.0f-6, config.nepochs))
    elseif config.lr_scheduler == :CONSTANT
        ParameterSchedulers.Stateful(ParameterSchedulers.Constant(config.eta))
    else
        throw(ArgumentError("`config.lr_scheduler` must be either `:COSINE` or `:CONSTANT`"))
    end

    return opt, sched
end

is_distributed() = FluxMPI.Initialized() && total_workers() > 1

_get_loggable_stats(::Nothing) = ()

function _get_loggable_stats(stats::NamedTuple)
    if is_distributed()
        arr = [stats.mean_nfe, stats.accuracy, stats.loss, stats.total_datasize]
        MPI.Reduce!(arr, +, 0, MPI.COMM_WORLD)
        return ((arr[1:3] ./ arr[4])..., stats.total_time)
    else
        return (
            stats.mean_nfe / stats.total_datasize,
            stats.accuracy / stats.total_datasize,
            stats.loss / stats.total_datasize,
            stats.total_time,
        )
    end
end

evaluate(model, ps, st, ::Nothing, device) = nothing

function evaluate(model, ps, st, dataloader, device)
    st_eval = EFL.testmode(st)
    matches, total_loss, total_datasize, total_nfe, total_time = 0, 0, 0, 0, 0
    for (x, y) in dataloader
        x = device(x)
        y = device(y)

        start_time = time()
        (ŷ, soln), _ = model(x, ps, st_eval)
        total_time += time() - start_time

        total_nfe += soln.nfe * size(x, ndims(x))
        total_loss += Flux.Losses.logitcrossentropy(ŷ, y) * size(x, ndims(x))
        matches += sum(argmax.(eachcol(cpu(ŷ))) .== Flux.onecold(cpu(y)))
        total_datasize += size(x, ndims(x))
    end
    return (loss=total_loss, accuracy=matches, mean_nfe=total_nfe, total_time=total_time, total_datasize=total_datasize)
end

function train_one_epoch(
    model,
    ps,
    st,
    loss_function,
    opt_state,
    dataloader,
    device,
    lg::PrettyTableLogger,
    econfig::ExperimentConfiguration,
    iteration_count::Int,
)
    total_time = 0

    for (x, y) in dataloader
        x = device(x)
        y = device(y)

        # Compute Loss + Backprop + Update
        start_time = time()

        (loss, ŷ, st, nfe), back = Flux.pullback(p -> loss_function(x, y, model, p, st), ps)
        gs, = back((one(loss), nothing, nothing, nothing))
        opt_state, ps = Optimisers.update!(opt_state, ps, gs)

        total_time += time() - start_time

        acc = sum(argmax.(eachcol(cpu(ŷ))) .== Flux.onecold(cpu(y))) / size(x, 4)

        iteration_count += 1
        st = econfig.pretrain_steps == iteration_count ? EFL.update_state(st, :fixed_depth, 0) : st
        if iteration_count % 25 == 0
            # Without this we might frequently run out of memory
            # Disabling CUDA Mempool with MPI seems to help but it
            # also slows down the overall code
            # Relieve GC Pressure
            relieve_gc_pressure((gs, ŷ, x, y))
            invoke_gc()
        end

        # Logging
        lg(; records=Dict("Train/Running/NFE" => nfe, "Train/Running/Loss" => loss, "Train/Running/Accuracy" => acc))
    end

    return ps, st, opt_state, iteration_count, (total_time=total_time,)
end

loss_function(e::ExperimentConfiguration, args...; kwargs...) = loss_function(e.model_config, args...; kwargs...)

function loss_function(c::ImageClassificationModelConfiguration; λ_skip=1.0f-2)
    function loss_function_closure(x, y, model, ps, st)
        (ŷ, soln), st_ = model(x, ps, st)
        loss = if c.model_type == :VANILLA
            Flux.Losses.logitcrossentropy(ŷ, y)
        else
            Flux.Losses.logitcrossentropy(ŷ, y) + λ_skip * Flux.Losses.mae(soln.u₀, Flux.Zygote.dropgrad(soln.z_star))
        end
        return loss, ŷ, st_, soln.nfe
    end
    return loss_function_closure
end

function train(
    model,
    ps,
    st,
    loss_function,
    opt,
    scheduler,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    device,
    nepochs,
    lg::PrettyTableLogger,
    econfig::ExperimentConfiguration;
)
    invoke_gc()
    # TODO: Saving model weights
    opt_state = Optimisers.setup(opt, ps)
    opt_state = is_distributed() ? FluxMPI.synchronize!(opt_state; root_rank=0) : opt_state
    iteration_count = 0

    st = econfig.pretrain_steps != 0 ? EFL.update_state(st, :fixed_depth, econfig.model_config.num_layers) : st

    for epoch in 1:nepochs
        # Train 1 epoch
        ps, st, opt_state, iteration_count, training_stats = train_one_epoch(
            model, ps, st, loss_function, opt_state, train_dataloader, device, lg, econfig, iteration_count
        )
        invoke_gc()

        # Evaluate
        val_stats = _get_loggable_stats(evaluate(model, ps, st, val_dataloader, device))
        invoke_gc()
        test_stats = _get_loggable_stats(evaluate(model, ps, st, test_dataloader, device))
        invoke_gc()

        lg(epoch, training_stats.total_time, val_stats..., test_stats...)

        # Run ParameterScheduler
        eta_new = ParameterSchedulers.next!(scheduler)
        opt_state = update_lr(opt_state, eta_new)
    end

    return ps, st, opt_state
end
