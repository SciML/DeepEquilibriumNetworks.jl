function evaluate(model, ps, st, dataloader, device)
    matches, total_loss, total_datasize, total_nfe = 0, 0, 0, 0
    for (x, y) in dataloader
        x = device(x)
        y = device(y)
        (ŷ, soln), st = model(x, ps, st)

        total_nfe += soln.nfe * size(x, ndims(x))
        total_loss += Flux.Losses.logitcrossentropy(ŷ, y) * size(x, ndims(x))
        matches += sum(argmax.(eachcol(ŷ)) .== Flux.onecold(cpu(y)))
        total_datasize += size(x, ndims(x))
    end
    return (
        (
            loss=total_loss / total_datasize,
            accuracy=matches / total_datasize,
            mean_nfe=total_nfe / total_datasize
        ),
        st
    )
end

function train_one_epoch()
end