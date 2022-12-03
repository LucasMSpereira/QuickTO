# Functions related to machine learning

#= k-fold cross-validation: number of folds in which training/validation
dataset will be split. In each of the k iterations, (k-1) folds will be used for
training, and the remaining fold will be used for validation =#
function crossVal(data, numFolds, activ, epochs, batch, opt, save)
  for fold in kfolds(shuffleobs(data); k = numFolds)
    model, modParams = buildModel(activ) # build model
    # number of times to go through entire training dataset,
    # considering k-fold cross-validation
    for epoch in 1:epochs
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

# calculate earlystop
function earlyStopCalc(validationHistory, earlyStopQuant)
  (validationHistory[end] - validationHistory[end - earlyStopQuant]) / validationHistory[end - earlyStopQuant] * 100
end

# Check for early stop: if validation loss didn't
# decrease enough in the last few validations
function earlyStopCheck(trainParams)
  if isa(trainParams, GANmetaData) # in case of GAN training
    # percentage drop in generator validation loss
    genValLossPercentDrop = earlyStopCalc(
      trainParams.lossesVals[:genValHistory],
      trainParams.trainConfig.earlyStopQuant
    )
    # percentage drop in generator validation loss
    discValLossPercentDrop = earlyStopCalc(
      trainParams.lossesVals[:discValHistory],
      trainParams.trainConfig.earlyStopQuant
    )
    # Return percentage drop and boolean that signals if training should stop
    return (
      genValLossPercentDrop,
      discValLossPercentDrop,
      prod([genValLossPercentDrop discValLossPercentDrop] .> -trainParams.trainConfig.earlyStopPercent)
    )
  else
    valLossPercentDrop = earlyStopCalc(trainParam.evaluations, trainParam.earlyStopQuant)
    # Return percentage drop and boolean that signals if training should stop
    return valLossPercentDrop, valLossPercentDrop > -trainParams.earlyStopPercent
  end
end

# train GANs with earlystopping
function earlyStopGANs(metaData)
  epoch = 0
  while true # loop in epochs
    epoch += 1 # count training epochs
    epoch == 1 && println("Epoch       Generator loss    Discriminator loss")
    # trains for one epoch
    GANepoch!(metaData, :train)
    # occasionally run validation epoch and check for early-stopping
    if epoch % metaData.trainConfig.validFreq == 0
      switchTraining(metaData, false) # disable model updating during validation
      # validation epoch returning avg losses for both NNs
      GANepoch!(metaData, :validate) |> metaData
      # print information about validation
      switchTraining(metaData, true) # reenable model updating after validation
      # after enough validations, start performing early-stop check
      if length(metaData.lossesVals[:genValHistory]) > metaData.trainConfig.earlyStopQuant
        # percentage drops in losses and boolean indicating to stop training
        genValLossPercentDrop, discValLossPercentDrop, earlyStopping = earlyStopCheck(metaData)
        # print validation and early-stop information
        GANprints(epoch, metaData; earlyStopVals = (genValLossPercentDrop, discValLossPercentDrop))
        if earlyStopping
          println("EARLY-STOPPING")
          saveGANs(metaData; finalSave = true)
          break
        end
      else # not checking for early-stop yet
        GANprints(epoch, metaData) # print just validation information
      end
    end
    # save occasional checkpoints of the model
    epoch % metaData.trainConfig.checkPointFreq == 0 && saveGANs(metaData)
  end
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

# train GANs with fixed number of epochs
function fixedEpochGANs(metaData)
  epoch = 0
  while epoch < metaData.trainConfig.epochs # loop in epochs
    epoch += 1 # count training epochs
    epoch == 1 && println("Epoch       Generator loss    Discriminator loss")
    GANepoch!(metaData, :train) # training epoch
    # occasionally run validation epoch
    if epoch % metaData.trainConfig.validFreq == 0
      switchTraining(metaData, false) # disable model updating during validation
      # validation epoch returning avg losses for both NNs
      GANepoch!(metaData, :validate) |> metaData
      switchTraining(metaData, true) # reenable model updating after validation
      GANprints(epoch, metaData) # print information about validation
    end
    # save occasional checkpoints of the models
    epoch % metaData.trainConfig.checkPointFreq == 0 && saveGANs(metaData)
  end
  saveGANs(metaData; finalSave = true) # save final models
end

# epoch of GAN usage, be it training, validation or test
# return avg. losses for epoch
function GANepoch!(metaData, goal)
  !in(goal, [:train, :validate, :test]) && error("GANepoch!() called with invalid 'goal'.")
  # initialize variables related to whole epoch
  genLossHist, discLossHist, batchCount = 0.0, 0.0, 0
  groupFiles = defineGroupFiles(metaData, goal)
  # loop in groups of files used for current split
  for group in groupFiles
    # get loader with data for current group
    currentLoader = GANdataLoader(metaData, goal, group)
    # each batch of current epoch
    for (genInput, FEAinfo, realTopology) in currentLoader
      batchCount += 1
      GC.gc(); CUDA.reclaim() # avoid GPU memory issues
      # use NNs, and get gradients and losses for current batch
      genGrads, genLossVal, discGrads, discLossVal = GANgrads(
        metaData.generator, metaData.discriminator, genInput, FEAinfo, realTopology
      )
      if goal == :train # update NNs parameters in case of training
        Flux.Optimise.update!(metaData.optInfo.genState, metaData.generator, genGrads[1])
        Flux.Optimise.update!(metaData.optInfo.discState, metaData.discriminator, discGrads[1])
      end
      # acumulate batch losses
      genLossHist += genLossVal; discLossHist += discLossVal
    end
  end
  # return avg losses for current epoch
  return genLossHist/batchCount, discLossHist/batchCount
end

# Test hyperparameter combinations with grid search
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
  BSON.@save "./networks/models/checkpoints/$(timeNow()).bson" cpu_model # save model
  gpu(MLmodel)
end

# In hyperGrid() function, save model and generate pdf report
# with training plot tests, and info when necessary
function saveModelAndReport!(k, activ, ch, params, comb, history, MLmodel, currentTest, testData, multiLossArch, FEparams, lossFun, optimizer)
  # Name used to save files related to current model
  currentModelName = "$k-$activ-$ch-"*sciNotation(minimum(params.evaluations), 7)
  # Create folder for current model
  mkpath("./networks/models/$currentModelName")
  cpu_model = cpu(MLmodel) # transfer model back to cpu
  BSON.@save "./networks/models/$currentModelName/$currentModelName.bson" cpu_model # save model
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

# Get outputs from generator and discriminator


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
    BSON.@save "./networks/models/$modelName/FEA-$(rand(1:99999)).bson" cpu_model
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
    BSON.@save "./networks/models/$(timeNow()).bson" cpu_model
  end
end