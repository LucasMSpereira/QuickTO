# Functions related to machine learning

# Epoch of batch training ML model. Then take optimizer step
function batchTrain!(trainDataLoader, mlModel, opt)
  for (x, y) in trainDataLoader
    grads = Flux.gradient(Flux.params(mlModel)) do
      Flux.mse(mlModel(gpu(x)), gpu(y))
    end
    # Optimization step for current batch
    Flux.Optimise.update!(opt, Flux.params(mlModel), grads)
  end
end

# Model structure
function convMaxPoolBN(kernel, activ, ch)
  # kernel size of 1-5
  module1 = Chain(
    BatchNorm(1),
    Conv(kernel, 1 => ch, activ), # outputsize logic: 1 + (L - k)
    MaxPool(kernel), # floor(L/k)
    BatchNorm(ch),
    Conv(kernel, ch => 2*ch, activ),
    MaxPool(kernel),
    BatchNorm(2*ch),
    flatten)
  m1size = prod(Flux.outputsize(module1, (50, 140, 1, 1)))
  module2 = Chain(Dense(m1size => m1size÷2), Dense(m1size÷2 => 8))
  return Chain(module1, module2) |> gpu
end

#= k-fold cross-validation: number of folds in which training/validation
dataset will be split. In each of the k iterations, (k-1) folds will be used for
training, and the remaining fold will be used for validation =#
function crossVal(data, numFolds, activ, epochs, batch, opt, save)
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
    meanEval = Flux.mae(cpu_model(fold[2][1]), fold[2][2])
    println("Fold validation: $(meanEval)")
    # Save model including mean evaluation loss in the BSON file's name
    if save == "eval"
      @save "./networks/models/$(replace(string(round( meanEval; digits = 6 )), "." => "-")).bson" cpu_model
    elseif save == "time"
      @save "./networks/models/$(timeNow()).bson" cpu_model
    else
      throw("Wrong 'save' specification in crossVal()")
    end
  end
end

# Evaluate (validation or test) model and print performance. To be called occasionally
function epochEval!(evalDataLoader, mlModel, trainParams, epoch, learnRate; test = false)
  Flux.trainmode!(mlModel, false)
  meanEvalLoss = zeros(ceil(Int, length(evalDataLoader.data.data.indices[end])/evalDataLoader.batchsize))
  # Go through validation dataset
  # Batches are used for GPU memory purposes
  count = 0
  for (x, y) in evalDataLoader
    count += 1
    meanEvalLoss[count] = Flux.mse(mlModel(gpu(x)), gpu(y))
  end
  trainmode!(mlModel, true)
  # If function was called in a validation context
  if !test
    # Keep history of evaluations
    trainParams.evaluations = vcat(trainParams.evaluations, mean(meanEvalLoss))
    # Print info
    if length(trainParams.evaluations) > 1
      @printf "Epoch %i     Δ(Validation loss): %.3e     Learning rate: %.0e     " epoch (trainParams.evaluations[end] - trainParams.evaluations[end-1]) learnRate
    else
      @printf "Epoch %i     Validation loss: %.3e     Learning rate: %.0e     " epoch trainParams.evaluations[end] learnRate
    end
    typeof(trainParams) == epochTrainConfig && println()
  else
    # If function was called in a test context
    return mean(meanEvalLoss)
  end
end

# Check for early stop: if validation loss didn't
# decrease enough in the last few validations
function earlyStopCheck(trainParams)
  currentValidLoss = trainParams.evaluations[end]
  pastValidLoss = trainParams.evaluations[end - trainParams.earlyStopQuant]
  # Return boolean that signals if training should stop
  valLossPercentDrop = (currentValidLoss - pastValidLoss)/pastValidLoss*100 
  return valLossPercentDrop, valLossPercentDrop > -trainParams.earlyStopPercent
end

# Test hyperparameter combinations (grid search)
function hyperGrid(architecture::Function, kernelSizes::Array{Int}, activFunctions, channels::Array{Int}, data, FEparams)
  # Number of combinations
  numCombs = length(kernelSizes)*length(activFunctions)*length(channels)
  println("\nTesting $numCombs combinations of hyperparameters.")
  comb = 0
  # Store history of test performances
  history = ["0" 0.0]
  # Test combinations
  for k in kernelSizes, activ in activFunctions, ch in channels
    comb += 1
    # Name used to save files related to current model
    currentModelName = "$k-$activ-$ch"
    println("\nCombination $comb/$numCombs - kernel size: ($k, $k)    activation function: $activ    channels: 1 => $ch => $(2*ch)")
    # build model with current combinations of hyperparameters
    model = architecture((k, k), activ, ch)
    params = earlyStopTrainConfig(15; earlyStopPercent = 0.5) # initialize training parameters
    # Train current model with early-stopping
    trainEarlyStop!(model, data[1:2]..., Flux.Optimise.RMSProp(1e-5), params; modelName = currentModelName, saveModel = false)
    # Avg. loss in test of current model
    currentTest = epochEval!(data[3], model, params, 1, 1e-5; test = true)
    # Save model if first combination of hyperparameters or new best performance in tests
    if comb == 1 || (currentTest < minimum(history[:, 2]))
      comb == 1 ? println("Saving first model and report...") : println("New best in test! Saving model and report...")
      comb > size(history, 2) ? (history = vcat(history, [currentModelName currentTest])) : (history[comb, 1] = currentModelName; history[comb, 2] = currentTest)
      hyperGridSave(currentModelName, params, 1e-5, data[3].data.data.parent, reshape(data[3].data.label.parent, (2, 4, :)), FEparams, model)
      cpu_model = cpu(model) # transfer model back to cpu
      @save "./networks/models/$currentModelName/$currentModelName.bson" cpu_model # save model
    else
      println("Model not saved.")
    end
  end
  return history
end

# test different learning rates
function testRates!(rates, trainConfigs, trianLoader, validateLoader, opt)
  # Multiple instances to store histories
  for (rate, parameters) in zip(rates, trainConfigs)
    model = buildModel(leakyrelu)
    trainEpochs!(model, trianLoader, validateLoader, opt(rate), parameters)
  end
  return trainConfigs
end

#= Train ML model with early stopping.
In predetermined intervals of epochs, evaluate the current
model and print validation loss.=#
function trainEarlyStop!(mlModel, trainDataLoader, validateDataLoader, opt, trainParams; modelName = timeNow(), saveModel = true)
  # Training epochs loop
  println("Early-stop training...")
  epoch = 1
  while true
    # In case of learning rate scheduling, apply decay at certain interval of epochs
    ((trainParams.schedule != 0) && (epoch % trainParams.schedule == 0)) && (opt.eta *= trainParams.decay)
    # Batch training and parameter update
    batchTrain!(trainDataLoader, mlModel, opt)
    # Evaluate model and print info at certain epoch intervals
    if epoch % trainParams.validFreq == 0
      epochEval!(validateDataLoader, mlModel, trainParams, epoch, opt.eta)
      # Early stopping
      if length(trainParams.evaluations) > trainParams.earlyStopQuant
        # Check for early stop criterion
        valLossPercentDrop, stopTraining = earlyStopCheck(trainParams)
        @printf "Early stop: %.1f%%/-%.1f%%\n" valLossPercentDrop trainParams.earlyStopPercent
        if stopTraining
          println("EARLY STOPPING")
          break
        end
      else
        println("No early stop check.")
      end
    end
    epoch += 1
  end
  if saveModel
    cpu_model = cpu(mlModel)
    @save "./networks/models/$modelName.bson" cpu_model
  end
end

#= Train ML model for certain number of epochs.
In predetermined intervals of epochs, evaluate the current
model and print validation loss.=#
function trainEpochs!(mlModel, trainDataLoader, validateDataLoader, opt, trainParams)
  # Epochs loop
  for epoch in 1:trainParams.epochs
    if epoch == 1
      @show opt.eta
    end
    # In case of learning rate scheduling, apply decay at certain interval of epochs
    ((trainParams.schedule != 0) && (epoch % trainParams.schedule == 0)) && (opt.eta *= trainParams.decay)
    # Batch training and parameter update
    batchTrain!(trainDataLoader, mlModel, opt)
    # Evaluate model and print info at certain epoch intervals
    epoch % trainParams.validFreq == 0 && epochEval!(validateDataLoader, mlModel, trainParams, epoch, opt.eta)
  end
  cpu_model = cpu(mlModel)
  @save "./networks/models/$(timeNow()).bson" cpu_model
end

mutable struct earlyStopTrainConfig
  validFreq::Int64 # Evaluation frequency in epochs
  decay::Float64 # Factor applied to current learning rate (in case of scheduling)
  schedule::Int64 # Learning rate adjustment interval in epochs. 0 for no scheduling
  evaluations::Array{Any} # History of evaluation losses
  # Interval of most recent validations used for early stop criterion
  earlyStopQuant::Int32
  # Minimal percentage drop in loss in the last "earlyStopQuant" validations
  earlyStopPercent::Float32
end
earlyStopTrainConfig(validFreq; decay=0.0, schedule=0, evaluations=[], earlyStopQuant=5, earlyStopPercent=1) = earlyStopTrainConfig(validFreq, decay, schedule, evaluations, earlyStopQuant, earlyStopPercent)

mutable struct epochTrainConfig
  epochs::Int64 # Total number of training epochs
  validFreq::Int64 # Evaluation frequency in epochs
  schedule::Int64 # Learning rate adjustment interval in epochs. 0 for no scheduling
  decay::Float64 # Factor applied to current learning rate (in case of scheduling)
  evaluations::Array{Any} # History of evaluation losses
end
epochTrainConfig(epochs, validFreq; schedule=0, decay=0.0, evaluations=[]) = epochTrainConfig(epochs, validFreq, schedule, decay, evaluations)