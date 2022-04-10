evaluate(model, ps, st, ::Nothing, device) = nothing

function evaluate(model, ps, st, dataloader, device)
    matches, total_loss, total_datasize, total_nfe, total_time = 0, 0, 0, 0, 0
    for (x, y) in dataloader
        x = device(x)
        y = device(y)

        start_time = time()
        (ŷ, soln), _ = model(x, ps, st)
        total_time += time() - start_time

        total_nfe += soln.nfe * size(x, ndims(x))
        total_loss += Flux.Losses.logitcrossentropy(ŷ, y) * size(x, ndims(x))
        matches += sum(argmax.(eachcol(ŷ)) .== Flux.onecold(cpu(y)))
        total_datasize += size(x, ndims(x))
    end
    return (
        loss=total_loss / total_datasize,
        accuracy=matches / total_datasize,
        mean_nfe=total_nfe / total_datasize,
        total_time=total_time,
    )
end

function train_one_epoch(model, ps, st, loss_function, opt_state, dataloader, device, lg::PrettyTableLogger)
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

        # Logging
        lg(; records=Dict("Train/Running/NFE" => nfe, "Train/Running/Loss" => loss, "Train/Running/Accuracy" => acc))
    end

    return ps, st, opt_state, (total_time=total_time,)
end

function loss_function(dataset::Symbol, model_type::Symbol)
    if dataset ∈ (:CIFAR10,)
        function loss_function_closure(x, y, model, ps, st)
            (ŷ, soln), st_ = model(x, ps, st)
            loss = if model_type == :vanilla
                Flux.Losses.logitcrossentropy(ŷ, y)
            else
                Flux.Losses.logitcrossentropy(ŷ, y) + Flux.Losses.mse(soln.u₀, soln.z_star)
            end
            return loss, ŷ, st_, soln.nfe
        end
        return loss_function_closure
    else
        throw(ArgumentError("$dataset - $model_type not yet supported"))
    end
end

function train(
    model,
    ps,
    st,
    loss_function,
    opt,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    device,
    nepochs,
    lg::PrettyTableLogger;
    distributed::Bool=false,
    cleanup_function=identity,
)
    # TODO: Saving model weights
    opt_state = Optimisers.setup(opt, ps)

    for epoch in 1:nepochs
        # Train 1 epoch
        ps, st, opt_state, training_stats = train_one_epoch(
            model, ps, st, loss_function, opt_state, train_dataloader, device, lg
        )
        cleanup_function()

        # Evaluate
        val_eval_stats = evaluate(model, ps, st, val_dataloader, device)
        cleanup_function()
        test_eval_stats = evaluate(model, ps, st, test_dataloader, device)
        cleanup_function()

        val_stats, test_stats = if distributed
            # TODO: Implement syncing the statistics
            error("Distributed Training not yet implemented")
        else
            (
                if val_eval_stats === nothing
                    ()
                else
                    (val_eval_stats.mean_nfe, val_eval_stats.accuracy, val_eval_stats.loss, val_eval_stats.total_time)
                end,
                if test_eval_stats === nothing
                    ()
                else
                    (test_eval_stats.mean_nfe, test_eval_stats.accuracy, test_eval_stats.loss, test_eval_stats.total_time)
                end,
            )
        end

        lg(epoch, training_stats.total_time, val_stats..., test_stats...)
    end

    return ps, st, opt_state
end
