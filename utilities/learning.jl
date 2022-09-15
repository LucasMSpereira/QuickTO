# Functions related to machine learning

# Epoch of batch training ML model. Then take optimizer step
function batchTrain!(trainDataLoader, mlModel, opt, lossFun)
  for (x, y) in trainDataLoader
    grads = Flux.gradient(Flux.params(mlModel)) do
      lossFun(mlModel(gpu(x)), gpu(y))
    end
    # Optimization step for current batch
    Flux.Optimise.update!(opt, Flux.params(mlModel), grads)
  end
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

# Check for early stop: if validation loss didn't
# decrease enough in the last few validations
function earlyStopCheck(trainParams)
  currentValidLoss = trainParams.evaluations[end]
  pastValidLoss = trainParams.evaluations[end - trainParams.earlyStopQuant]
  # Return boolean that signals if training should stop
  valLossPercentDrop = (currentValidLoss - pastValidLoss)/pastValidLoss*100 
  return valLossPercentDrop, valLossPercentDrop > -trainParams.earlyStopPercent
end

# Evaluate (validation or test) model and print performance. To be called occasionally
function epochEval!(evalDataLoader, mlModel, trainParams, epoch, lossFun; test = false)
  Flux.trainmode!(mlModel, false)
  meanEvalLoss = zeros(ceil(Int, length(evalDataLoader.data.data.indices[end])/evalDataLoader.batchsize))
  # Go through validation dataset
  # Batches are used for GPU memory purposes
  count = 0
  for (x, y) in evalDataLoader
    count += 1
    meanEvalLoss[count] = lossFun(mlModel(gpu(x)), gpu(y))
  end
  trainmode!(mlModel, true)
  # If function was called in a validation context
  if !test
    # Keep history of evaluations
    trainParams.evaluations = vcat(trainParams.evaluations, mean(meanEvalLoss))
    # Print info
    if length(trainParams.evaluations) > 1
      @printf "Epoch %i     Δ(Validation loss): %.3e     " epoch (trainParams.evaluations[end] - trainParams.evaluations[end-1])
    else
      @printf "Epoch %i     Validation loss: %.3e     " epoch trainParams.evaluations[end]
    end
    typeof(trainParams) == epochTrainConfig && println()
  else
    # If function was called in a test context
    return mean(meanEvalLoss)
  end
end

# prepare data loaders for loadCNN training/validation/test using displacements
function getDisploaders(disp, forces, numPoints, separation, batch; multiOutputs = false)
  if numPoints != 0 # use all data or certain number of points
    if !multiOutputs
      dispTrain, dispValidate, dispTest = splitobs((disp[:, :, :, 1:numPoints], forces[:, 1:numPoints]); at = separation, shuffle = true)
    else
      dispTrain, dispValidate, dispTest = splitobs(
        (disp[:, :, :, 1:numPoints], (forces[1][:, 1:numPoints], forces[2][:, 1:numPoints], forces[3][:, 1:numPoints], forces[4][:, 1:numPoints]));
        at = separation, shuffle = true)
    end
  else
    dispTrain, dispValidate, dispTest = splitobs((disp, forces); at = separation, shuffle = true)
  end
  return ( # DataLoader serves to iterate in mini-batches of the training data
    DataLoader((data = dispTrain[1], label = dispTrain[2]); batchsize = batch, parallel = true),
    DataLoader((data = dispValidate[1], label = dispValidate[2]); batchsize = batch, parallel = true),
    DataLoader((data = dispTest[1], label = dispTest[2]); batchsize = batch, parallel = true))
end

# prepare data loaders for stressCNN training/validation/test
function getVMloaders(vm, forces, numPoints, separation, batch; multiOutputs = false)
  # use all data or certain number of points
  if numPoints != 0
    if !multiOutputs
      vmTrain, vmValidate, vmTest = splitobs((vm[:, :, :, 1:numPoints], forces[:, 1:numPoints]); at = separation, shuffle = true)
    else
      vmTrain, vmValidate, vmTest = splitobs(
        (vm[:, :, :, 1:numPoints], (forces[1][:, 1:numPoints], forces[2][:, 1:numPoints], forces[3][:, 1:numPoints], forces[4][:, 1:numPoints]));
        at = separation, shuffle = true)
    end
  else
    vmTrain, vmValidate, vmTest = splitobs((vm, forces); at = separation, shuffle = true)
  end
  # DataLoader serves to iterate in mini-batches of the training data
  vmTrainLoader = DataLoader((data = vmTrain[1], label = vmTrain[2]); batchsize = batch, parallel = true);
  vmValidateLoader = DataLoader((data = vmValidate[1], label = vmValidate[2]); batchsize = batch, parallel = true);
  vmTestLoader = DataLoader((data = vmTest[1], label = vmTest[2]); batchsize = batch, parallel = true);
  return vmTrainLoader, vmValidateLoader, vmTestLoader
end

# Test hyperparameter combinations (grid search)
function hyperGrid(
  architecture::Function, kernelSizes::Array{Int}, activFunctions,
  channels::Array{Int}, data, FEparams, lossFun, optimizer; earlyStop = 0.0, multiLossArch = false
)
  # Number of combinations
  numCombs = length(kernelSizes)*length(activFunctions)*length(channels)
  println("\nTesting $numCombs combinations of hyperparameters.")
  comb = 0
  # Store history of test performances
  history = ["0" 0.0]
  # Test combinations of hyperparameters (grid-search)
  for k in kernelSizes, activ in activFunctions, ch in channels
    comb += 1
    println("\nCombination $comb/$numCombs - kernel size: ($k, $k)    activation function: $activ    channels: $ch")
    model = architecture((k, k), activ, ch) # build model with current combinations of hyperparameters
    params = earlyStopTrainConfig(15; earlyStopPercent = earlyStop) # initialize training configuration
    # Train current model with early-stopping
    trainEarlyStop!(model, data[1:2]..., optimizer, params, lossFun; saveModel = false)
    currentTest = epochEval!(data[3], model, params, 1, lossFun; test = true) # Avg. loss in test of current model
    # Save model if first combination of hyperparameters or new best performance in tests so far
    if comb == 1 || (currentTest < minimum(history[:, 2]))
      # save model and generate pdf report with training plot tests, and info
      saveModelAndReport!(k, activ, ch, params, comb, history, model, currentTest, data[3], multiLossArch, FEparams, lossFun, optimizer)
    else
      println("Model not saved.")
    end
  end
  return history
end

# Generate pdf with validation history and test plots for hyperparameter grid search
function hyperGridSave(currentModelName, trainParams, vm, forceData, FEparams, MLmodel, lossFun, optimizer, multiLossArch = false)
  # PDF with validation history plot
  plotLearnTries([trainParams], [optimizer.eta]; drawLegend = false, name = currentModelName*" valLoss", path = "./networks/models/$currentModelName")
  # PDF with list of hyperparameters
  parameterList(MLmodel, optimizer, lossFun, "./networks/models/$currentModelName"; multiLossArch)
  # Create test plots and unite all PDF files in folder
  loadCNNtestPlots(20, "./networks/models/$currentModelName", vm, forceData, currentModelName, FEparams, MLmodel, lossFun)
end

# occasionally save model
function intermediateSave(params, MLmodel)
  println("Checkpoint save")
  # Name used to save files related to current model
  currentModelName = "$k-$activ-$ch-"*sciNotation(minimum(params.evaluations), 7)
  # Create folder for current model
  mkpath("./networks/models/$currentModelName")
  cpu_model = cpu(MLmodel) # transfer model back to cpu
  @save "./networks/models/$currentModelName/$currentModelName.bson" cpu_model # save model
end

# Loss for NN architectures with more than one output
function multiLoss(output, target; lossFun)
  # Loss compares each prediction/target pair
  # From function getLoaders with multi-output: x positions,
  # y positions, components of first load, components of second load
  return mean([lossFun(modelOut, truth) for (modelOut, truth) in zip(output, target)])
end

# Generate pdf report from saved model, including test
# plots and architecture description
function reportFromSavedModel(pathModel, pathReport, opt, lossFun, multiOutput, vmTestLoader, FEAparams)
  @load pathModel cpu_model # load model from bson file
  gpu_model = gpu(cpu_model) # transfer model to gpu
  # PDF with list of hyperparameters
  parameterList(gpu_model, opt, lossFun, pathReport; multiLossArch = multiOutput)
  # shape of test targets. change in case of multiple outputs
  testTargets = shapeTargetStressCNN(multiOutput, vmTestLoader)
  # Create test plots and unite all PDF files in folder
  stressCNNtestPlots(
    20, pathReport, vmTestLoader.data.data.parent, testTargets,
    "reportFromSavedModel", FEAparams, gpu_model, lossFun
  )
end

# In hyperGrid() function, save model and generate pdf report
# with training plot tests, and info when necessary
function saveModelAndReport!(k, activ, ch, params, comb, history, MLmodel, currentTest, testData, multiLossArch, FEparams, lossFun, optimizer)
  # Name used to save files related to current model
  currentModelName = "$k-$activ-$ch-"*sciNotation(minimum(params.evaluations), 7)
  # Create folder for current model
  mkpath("./networks/models/$currentModelName")
  cpu_model = cpu(MLmodel) # transfer model back to cpu
  @save "./networks/models/$currentModelName/$currentModelName.bson" cpu_model # save model
  comb == 1 ? println("Saving first model and report...") : println("New best in test! Saving model and report...")
  # Store test score in "history" matrix
  comb > size(history, 1) ? (history = vcat(history, [currentModelName currentTest])) : (history[comb, 1] = currentModelName; history[comb, 2] = currentTest)
  # Determine shape of test targets. Changes if using multi-output models
  testTargets = shapeTargetStressCNN(multiLossArch, testData)
  # Generate pdf with validation history and visualization of tests
  hyperGridSave(currentModelName, params, testData.data.data.parent, testTargets, FEparams, MLmodel, lossFun, optimizer, multiLossArch)
end

#= Determine shape of test targets when generating report
for stress CNN in hiperGrid() function. Shape changes
if multi-output models are used =#
function shapeTargetStressCNN(multiLossArch::Bool, testData)
  if multiLossArch
    testTargets = zeros(Float32, (2, 4, size(testData.data.label[1], 2)))
    for sample in 1:size(testData.data.label[1], 2)
      [testTargets[:, col, sample] .= testData.data.label[col][:, sample] for col in 1:4]
    end
  else
    testTargets = reshape(testData.data.label.parent,(2, 4, :))
  end
  return testTargets
end

# test different learning rates and generate plot
function testRates!(rates, trainConfigs, trianLoader, validateLoader, opt, architecture, modelParams, lossFun)
  # Multiple instances to store histories
  for (currentRate, parameters) in zip(rates, trainConfigs)
    trainEpochs!(architecture(modelParams...), trianLoader, validateLoader, opt(currentRate), parameters, lossFun)
  end
  plotLearnTries(trainConfigs, rates)
end

#= Train ML model with early stopping.
In predetermined intervals of epochs, evaluate
the current model and print validation loss.=#
function trainEarlyStop!(mlModel, trainDataLoader, validateDataLoader, opt, trainParams, lossFun; modelName = timeNow(), saveModel = true)
  # Training epochs loop
  epoch = 1
  while true
    # In case of learning rate scheduling, apply decay at certain interval of epochs
    if (trainParams.schedule != 0) && (epoch % trainParams.schedule == 0)
      opt.eta *= trainParams.decay
      println("New learning rate: ", sciNotation(opt.eta, 1))
    end
    # Batch training and parameter update
    batchTrain!(trainDataLoader, mlModel, opt, lossFun)
    # Evaluate model and print info at certain epoch intervals
    if epoch % trainParams.validFreq == 0
      epochEval!(validateDataLoader, mlModel, trainParams, epoch, lossFun)
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
function trainEpochs!(mlModel, trainDataLoader, validateDataLoader, opt, trainParams, lossFun; save = false)
  # Epochs loop
  for epoch in 1:trainParams.epochs
    # In case of learning rate scheduling, apply decay at certain interval of epochs
    if (trainParams.schedule != 0) && (epoch % trainParams.schedule == 0)
      opt.eta *= trainParams.decay
      println("New learning rate: ", sciNotation(opt.eta, 1))
    end
    # Batch training and parameter update
    batchTrain!(trainDataLoader, mlModel, opt, lossFun)
    # Evaluate model and print info at certain epoch intervals
    epoch % trainParams.validFreq == 0 && epochEval!(validateDataLoader, mlModel, trainParams, epoch, lossFun)
  end
  if save
    cpu_model = cpu(mlModel)
    @save "./networks/models/$(timeNow()).bson" cpu_model
  end
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
earlyStopTrainConfig(validFreq; decay=0.0, schedule=0, evaluations=[], earlyStopQuant=3, earlyStopPercent=1) = earlyStopTrainConfig(validFreq, decay, schedule, evaluations, earlyStopQuant, earlyStopPercent)

mutable struct epochTrainConfig
  epochs::Int64 # Total number of training epochs
  validFreq::Int64 # Evaluation frequency in epochs
  schedule::Int64 # Learning rate adjustment interval in epochs. 0 for no scheduling
  decay::Float64 # Factor applied to current learning rate (in case of scheduling)
  evaluations::Array{Any} # History of evaluation losses
end
epochTrainConfig(epochs, validFreq; schedule=0, decay=0.0, evaluations=[]) = epochTrainConfig(epochs, validFreq, schedule, decay, evaluations)