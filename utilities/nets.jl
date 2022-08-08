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
function buildModel(activ)
  return Chain(
    BatchNorm(1),
    Conv((3, 3), 1 => 8, activ),
    MaxPool((3,3)),
    BatchNorm(8),
    Conv((5, 5), 8 => 16, activ),
    MaxPool((3,3)),
    BatchNorm(16),
    flatten,
    Dense(896 => 450),
    Dense(450 => 8),
  ) |> gpu
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
      BSON.@save "./networks/models/$(replace(string(round( meanEval; digits = 6 )), "." => "-")).bson" cpu_model
    elseif save == "time"
      BSON.@save "./networks/models/$(timeNow()).bson" cpu_model
    else
      throw("Wrong 'save' specification in crossVal()")
    end
  end
end

# Evaluate model and print performance. To be called occasionally
function epochEval!(validateDataLoader, mlModel, trainParams, epoch, learnRate)
  meanEvalLoss = []
  # Go through validation dataset
  # Batches are used for GPU memory purposes
  for (x, y) in validateDataLoader
    meanEvalLoss = vcat(meanEvalLoss, Flux.mse(mlModel(gpu(x)), gpu(y)))
  end
  # Keep history of evaluations
  trainParams.evaluations = vcat(trainParams.evaluations, mean(meanEvalLoss))
  # Print info
  if length(trainParams.evaluations) > 1
    @printf "Epoch %i\tΔ(Validation loss): %.6e\tLearning rate: %.0e\t" epoch (trainParams.evaluations[end] - trainParams.evaluations[end-1]) learnRate
  else
    @printf "Epoch %i\tValidation loss: %.6e\tLearning rate: %.0e\t" epoch trainParams.evaluations[end] learnRate
  end
  typeof(trainParams) == epochTrainConfig && println()
end

mutable struct epochTrainConfig
  epochs::Int64 # Total number of training epochs
  schedule::Int64 # Learning rate adjustment interval in epochs. 0 for no scheduling
  decay::Float64 # Factor applied to current learning rate (in case of scheduling)
  evalFreq::Int64 # Evaluation frequency in epochs
  evaluations::Array{Any} # History of evaluation losses
end
epochTrainConfig(
  ; epochs, schedule, decay, evalFreq, evaluations
) = epochTrainConfig(epochs, schedule, decay, evalFreq, evaluations)

# Check for early stop: percentage increase relative to
# minimum validation loss so far
function earlyStopCheck(trainParams)
  minValidLoss = minimum(trainParams.evaluations)
  return (trainParams.evaluations[end] - minValidLoss) / minValidLoss * 100
end

mutable struct earlyStopTrainConfig
  schedule::Int64 # Learning rate adjustment interval in epochs. 0 for no scheduling
  decay::Float64 # Factor applied to current learning rate (in case of scheduling)
  evalFreq::Int64 # Evaluation frequency in epochs
  evaluations::Array{Any} # History of evaluation losses
  # Percentage increase relative to minimum evaluation loss that triggers early stopping
  earlyStop::Float32
end
earlyStopTrainConfig(
  ; schedule, decay, evalFreq, evaluations, earlyStop
) = earlyStopTrainConfig(schedule, decay, evalFreq, evaluations, earlyStop)

function mlBOHB()
  function f(k, channel, activ)
    global dd += 1
    model = Chain(
      Conv((k, k), 1 => channel, activ)
    )
    obj = mean(model(mat))
    println("$dd $channel $activ $k $obj")
    return obj
  end
  mat = rand(50, 50, 1, 1)
  begin
    dd = 0
    HOHB = @hyperopt for i = 10,
        # List of possibilities of parameters
        sampler = Hyperband(R=50, η=3, inner = BOHB()),
        channel = [1, 2],
        activ = [relu, selu, swish],
        k = [2, 3]
        
      ob = vec([f(k, channel, activ)])

    end
  end
end

# test different learning rates
function testRates!(rates, trainConfigs)
  for (rate, parameters) in zip(rates, trainConfigs)
    stressCNN = buildModel(leakyrelu)
    trainEpochs!(stressCNN, vmTrainLoader, vmValidateLoader, Adam(rate), parameters)
  end
end

#= Train ML model with early stopping.
In predetermined intervals of epochs, evaluate the current
model and print validation loss.=#
function trainEarlyStop!(mlModel, trainDataLoader, validateDataLoader, opt, trainParams)
  # Training epochs loop
  @show opt.eta
  epoch = 1
  while true
    # In case of learning rate scheduling, apply decay at certain interval of epochs
    ((trainParams.schedule != 0) && (epoch % trainParams.schedule == 0)) && (opt.eta *= trainParams.decay)
    # Batch training and parameter update
    batchTrain!(trainDataLoader, mlModel, opt)
    # Evaluate model and print info at certain epoch intervals
    epoch % trainParams.evalFreq == 0 && epochEval!(validateDataLoader, mlModel, trainParams, epoch, opt.eta)
    # Early stopping
    if length(trainParams.evaluations) > 0
      esCriterion = earlyStopCheck(trainParams)
      epoch % trainParams.evalFreq == 0 && @printf "Early stop: %.1f%%/%.1f%%\n" esCriterion trainParams.earlyStop
      if esCriterion > trainParams.earlyStop
        println("EARLY STOPPING")
        break
      end
    end
    epoch += 1
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
    epoch % trainParams.evalFreq == 0 && epochEval!(validateDataLoader, mlModel, trainParams, epoch, opt.eta)
  end
end

