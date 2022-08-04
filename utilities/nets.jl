# Functions related to machine learning

# Model structure
function buildModel(activ)
  stressCNN = Chain(
    Conv((3, 3), 1 => 1, activ; stride = 3),
    Conv((3, 3), 1 => 2, activ),
    Conv((3, 3), 2 => 4, activ),
    Conv((5, 5), 4 => 8, activ),
    Conv((3, 3), 8 => 16, activ),
    Conv((3, 3), 16 => 32, activ),
    flatten,
    Dense(4352 => 2000),
    Dense(2000 => 500),
    Dense(500 => 8),
  ) |> gpu
  return stressCNN, Flux.params(stressCNN)
end

#= k-fold cross-validation: number of folds in which training/validation
dataset will be split. In each of the k iterations, (k-1) folds will be used for
training, and the remaining fold will be used for validation =#
function crossVal(data, numFolds, activ, epochs, batch, opt)
  for fold in kfolds(shuffleobs(data); k = numFolds)
    model, modParams = buildModel(activ) # build model
    # number of times to go through entire training dataset,
    # considering k-fold cross-validation
    @showprogress "Epoch" for epoch in 1:epochs
      trainLoader = DataLoader((data = fold[1][1], label = fold[1][2]); batchsize = batch, parallel = true)
      for (x, y) in trainLoader
        grads = Flux.gradient(modParams) do
          Flux.mae(model(gpu(x)), gpu(y))
        end
        Flux.Optimise.update!(opt, modParams, grads)
      end
    end
    cpu_model = cpu(model) # copy model of current fold to cpu
    # Mean of prediction loss in validation set of current fold
    meanEval = mean(Flux.mae(cpu_model(fold[2][1]), fold[2][2]))
    println("Fold validation: $(meanEval)")
    # Save model including mean evaluation loss in the BSON file's name
    # BSON.@save "model-$(replace(string(round( meanEval; digits = 6 )), "." => "-")).bson" cpu_model
    # BSON.@save "model-$(replace(string(ceil(now(), Dates.Second)), ":"=>"-")).bson" cpu_model
  end
end