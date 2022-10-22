# Functions related to machine learning

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

# Custom loss function for mulitple outputs in predicting loads from displacements
function customLoss1(x, y)
  # x: model output from one batch, y: batch of labels
  batchLoss = []
  for sampleInBatch in 1:size(y[1], 2) # iterate inside batch
    batchLoss = vcat(batchLoss, [(x[i][:, sampleInBatch] - y[i][:, sampleInBatch]) .^ 2 for i in axes(x)[1]] .|> mean |> mean)
  end
  return convert(Float32, mean(batchLoss)) |> gpu
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
function epochEval!(evalDataLoader, mlModel, trainParams, epoch, lossFun; test = false, FEAloss = false)
  Flux.trainmode!(mlModel, false)
  # Go through validation dataset
  # Batches are used for GPU memory purposes
  if FEAloss
    meanEvalLoss = batchEvalFEAloss!(evalDataLoader, mlModel)
  else
    meanEvalLoss = batchEval!(evalDataLoader, lossFun, mlModel)
  end
  trainmode!(mlModel, true)
  if !test # If function was called in a validation context
    # Keep history of evaluations
    trainParams.evaluations = vcat(trainParams.evaluations, meanEvalLoss)
    if length(trainParams.evaluations) > 1 # Print info
      @printf "Epoch %i     Δ(Validation loss): %.3e     " epoch (trainParams.evaluations[end] - trainParams.evaluations[end-1])
    else
      @printf "Epoch %i     Validation loss: %.3e     " epoch trainParams.evaluations[end]
    end
    typeof(trainParams) == epochTrainConfig && println()
  else # If function was called in a test context
    return meanEvalLoss
  end
end

#= FEAloss pipeline: further train model pre-trained by hyperGrid().
Now the loss calculates the new displacements resulting from the forces
predicted by the model. Then it compares the new displacements against
the old ones. Now the difference between displacement fields will be minimized. =#
function FEAlossPipeline(model, data, FEparams, lossFun, optimizer, modelName; earlyStop = 0.0)
  params = earlyStopTrainConfig(15; decay = 0.5, schedule = 100, earlyStopPercent = earlyStop) # initialize training configuration
  # Train model with early-stopping
  trainEarlyStop!(model, data[1:2]..., optimizer, params, lossFun; saveModel = true, FEAloss = true, modelName = modelName)
  Flux.trainmode!(model, false) # disable parameter training
  # save model and generate pdf report with training plot, tests, and info
  hyperGridSave(modelName, params, data[3], FEparams, model, lossFun, optimizer; multiLossArch = true)
  Flux.trainmode!(model, true) # reenable parameter training
end

# prepare data loaders for loadCNN training/validation/test using displacements
function getDisploaders(disp, forces, numPoints, separation, batch; multiOutputs = false)
  if numPoints != 0 # use all data or certain number of points
    if !multiOutputs
      dispTrain, dispValidate, dispTest = splitobs((disp[:, :, :, 1:numPoints], forces[:, 1:numPoints]); at = separation, shuffle = true)
    else
      dispTrain, dispValidate, dispTest = splitobs((disp[:, :, :, 1:numPoints],
          (forces[1][:, 1:numPoints], forces[2][:, 1:numPoints],
          forces[3][:, 1:numPoints], forces[4][:, 1:numPoints]));
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

# prepare data loaders for loadCNN training/validation/test in FEAloss pipeline
function getFEAlossLoaders(
  disp::Array{Float32, 4}, sup::Array{Float32, 3}, vf::Array{Float32, 1},
  force, numPoints::Int64, separation, batch::Int64
)
  # split dataset into training, validation, and testing
  numPoints == 0 && (numPoints = length(vf))
  numPoints < 0 && (error("Negative number of samples in getFEAlossLoaders."))
  FEAlossTrain, FEAlossValidate, FEAlossTest = splitobs(
    (
      disp[:, :, :, 1:numPoints],
      sup[:, :, 1:numPoints], vf[1:numPoints],
      (force[1][:, 1:numPoints], force[2][:, 1:numPoints],
      force[3][:, 1:numPoints], force[4][:, 1:numPoints])
    ); at = separation, shuffle = true)
  # return loaders for each stage
  return (DataLoader(FEAlossTrain; batchsize = batch, parallel = true),
    DataLoader(FEAlossValidate; batchsize = batch, parallel = true),
    DataLoader(FEAlossTest; batchsize = batch, parallel = true))
end

# prepare data loaders for loadCNN training/validation/test
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
  println("\nTesting $numCombs combination(s) of hyperparameters.")
  comb = 0
  # Store history of test performances
  history = ["0" 0.0]
  # Test combinations of hyperparameters (grid-search)
  for k in kernelSizes, activ in activFunctions, ch in channels
    comb += 1
    println("\nCombination $comb/$numCombs - kernel size: ($k, $k)    activation function: $activ    channels: $ch")
    model = architecture((k, k), activ, ch) # build model with current combinations of hyperparameters
    params = earlyStopTrainConfig(15; decay = 0.5, schedule = 100, earlyStopPercent = earlyStop) # initialize training configuration
    # Train current model with early-stopping
    trainEarlyStop!(model, data[1:2]..., optimizer, params, lossFun; saveModel = false)
    currentTest = epochEval!(data[3], model, params, 1, lossFun; test = true) # Avg. loss in test of current model
    # Save model if first combination of hyperparameters or new best performance in tests so far
    if comb == 1 || (currentTest < minimum(history[:, 2]))
      Flux.trainmode!(model, false) # disable parameter training
      # save model and generate pdf report with training plot, tests, and info
      saveModelAndReport!(k, activ, ch, params, comb, history, model, currentTest, data[3], multiLossArch, FEparams, lossFun, optimizer)
      Flux.trainmode!(model, true) # reenable parameter training
    else
      println("Model not saved.")
    end
  end
  return history
end

# Generate pdf with validation history and test plots for hyperparameter grid search
function hyperGridSave(currentModelName, trainParams, testLoader, FEparams, MLmodel, lossFun, optimizer; multiLossArch = false)
  # PDF with validation history plot
  plotLearnTries([trainParams], [optimizer.eta]; drawLegend = false, name = "a"*currentModelName*" valLoss", path = "./networks/models/$currentModelName")
  # PDF with list of hyperparameters
  parameterList(MLmodel, optimizer, lossFun, "./networks/models/$currentModelName"; multiLossArch)
  # Create test plots and unite all PDF files in folder
  dispCNNtestPlots(20, "./networks/models/$currentModelName", testLoader, currentModelName, FEparams, MLmodel, lossFun)
end

# occasionally save model
function intermediateSave(MLmodel)
  println("Checkpoint save")
  cpu_model = cpu(MLmodel) # transfer model to cpu
  @save "./networks/models/checkpoints/$(timeNow()).bson" cpu_model # save model
  gpu(MLmodel)
end

# Loss for NN architectures with more than one output
function multiLoss(output, target; lossFun)
  # Compares each prediction/target pair from the
  # function getLoaders() with multi-output: x positions,
  # y positions, components of first load, components of second load
  return mean([lossFun(modelOut, truth) for (modelOut, truth) in zip(output, target)])
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
  # Generate pdf with validation history and visualization of tests
  hyperGridSave(currentModelName, params, testData, FEparams, MLmodel, lossFun, optimizer, multiLossArch)
end

#= Determine shape of test targets when generating report
for stress CNN in hiperGrid() function. Shape changes
if multi-output models are used =#
function shapeTargetloadCNN(multiLossArch::Bool, testData)
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
function trainEarlyStop!(
  mlModel, trainDataLoader, validateDataLoader, opt, trainParams, lossFun;
  modelName = timeNow(), saveModel = true, FEAloss = false
)
  epoch = 1
  while true
    # In case of learning rate scheduling, apply decay at certain interval of epochs
    if (trainParams.schedule != 0) && (epoch % trainParams.schedule == 0)
      opt.eta *= trainParams.decay
      println("New learning rate: ", sciNotation(opt.eta, 1))
    end
    # Batch training and parameter update
    if FEAloss
      batchTrainFEAloss!(trainDataLoader, mlModel, opt)
    else
      batchTrain!(trainDataLoader, mlModel, opt, lossFun)
    end
    # Evaluate model and print info at certain epoch intervals
    if epoch % trainParams.validFreq == 0
      epochEval!(validateDataLoader, mlModel, trainParams, epoch, lossFun; FEAloss = FEAloss)
      if length(trainParams.evaluations) > trainParams.earlyStopQuant # Early stopping
        valLossPercentDrop, stopTraining = earlyStopCheck(trainParams) # Check for early stop criterion
        @printf "Early stop: %.1f%%/-%.1f%%\n" valLossPercentDrop trainParams.earlyStopPercent
        if stopTraining
          println("EARLY STOPPING")
          break
        end
      else
        println("No early stop check.")
      end
    end
    epoch % 300 == 0 && intermediateSave(mlModel) # intermediate save checkpoint
    epoch += 1
  end
  if saveModel
    cpu_model = cpu(mlModel)
    @save "./networks/models/$modelName/FEA-$(rand(1:99999)).bson" cpu_model
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
  decay::Float64 # If scheduling, periodically multiply learning rate by this value
  schedule::Int64 # Learning rate adjustment interval in epochs. 0 for no scheduling
  evaluations::Array{Any} # History of evaluation losses
  # Interval of most recent validations used for early stop criterion
  earlyStopQuant::Int32
  # Minimal percentage drop in loss in the last "earlyStopQuant" validations
  earlyStopPercent::Float32
end
earlyStopTrainConfig(
  validFreq; decay = 0.0, schedule = 0, evaluations = [], earlyStopQuant = 3, earlyStopPercent = 1
) = earlyStopTrainConfig(validFreq, decay, schedule, evaluations, earlyStopQuant, earlyStopPercent)

mutable struct epochTrainConfig
  epochs::Int64 # Total number of training epochs
  validFreq::Int64 # Evaluation frequency in epochs
  schedule::Int64 # Learning rate adjustment interval in epochs. 0 for no scheduling
  decay::Float64 # If scheduling, periodically multiply learning rate by this value
  evaluations::Array{Any} # History of evaluation losses
end
epochTrainConfig(
  epochs, validFreq; schedule = 0, decay = 0.0, evaluations = []
) = epochTrainConfig(epochs, validFreq, schedule, decay, evaluations)