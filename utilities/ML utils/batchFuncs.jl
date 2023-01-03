# Functions that loop in batches of the dataset, calculate gradients,
# and update ML model parameters

# Batch of inputs during validation/testing
function batchEval!(evalDataLoader, lossFun, mlModel)
  meanEvalLoss = zeros(ceil(Int, length(evalDataLoader.data.data.indices[end])/evalDataLoader.batchsize))
  count = 0
  for (x, y) in evalDataLoader # each batch
    count += 1
    meanEvalLoss[count] = lossFun(x |> gpu |> mlModel |> cpu, y)
  end
  return meanEvalLoss
end

# Batch of inputs during validation/testing. Adapted to FEAloss pipeline
function batchEvalFEAloss!(evalDataLoader, mlModel)
  meanEvalLoss = zeros(ceil(Int, length(evalDataLoader.data[1].indices[end])/evalDataLoader.batchsize))
  count = 0
  for (trueDisps, sup, vf, force) in evalDataLoader # each batch
    count += 1
    meanEvalLoss[count] = FEAloss(trueDisps |> gpu |> mlModel |> cpu, trueDisps, sup, vf, force)
  end
  return mean(meanEvalLoss)
end

# Epoch of batch training ML model. Then take optimizer step
function batchTrain!(trainDataLoader, mlModel, opt, lossFun)
  for (x, y) in trainDataLoader # each batch
    grads = Flux.gradient(Flux.params(mlModel)) do
      lossFun(x |> gpu |> mlModel |> cpu, y)
    end
    # Optimization step for current batch
    Flux.Optimise.update!(opt, Flux.params(mlModel), grads)
  end
end

# Epoch of batch training ML model. Then take optimizer step
# Adapted to FEAloss pipeline
function batchTrainFEAloss!(trainDataLoader, mlModel, opt)
  for (trueDisps, sup, vf, force) in trainDataLoader # each batch
    grads = Flux.gradient(
      () -> FEAloss(trueDisps |> gpu |> mlModel |> cpu, trueDisps, sup, vf, force),
      Flux.params(mlModel))
    # Optimization step for current batch
    Flux.Optimise.update!(opt, Flux.params(mlModel), grads)
  end
end