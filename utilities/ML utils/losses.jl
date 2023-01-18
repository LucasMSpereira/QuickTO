# Functions used in loss calculations for ML

# Custom loss function for mulitple outputs in predicting loads from displacements
function customLoss1(x, y)
  # x: model output from one batch, y: batch of labels
  batchLoss = []
  for sampleInBatch in 1:size(y[1], 2) # iterate inside batch
    batchLoss = vcat(batchLoss, [(x[i][:, sampleInBatch] - y[i][:, sampleInBatch]) .^ 2 for i in axes(x)[1]] .|> mean |> mean)
  end
  return convert(Float32, mean(batchLoss)) |> gpu
end

# use NNs and batch data to obtain gradients and losses
function GANgrads(
  metaData_::GANmetaData, goal::Symbol, genInput::Array{Float32},
  FEAinfo::Array{Float32}, realTopology::Array{Float32}
)::Tuple{Tuple{Any}, Float32, Tuple{Any}, Float32}
  logBatch = rand() < 0.1 # choose if this batch will be logged
  gen = getGen(metaData_); disc = getDisc(metaData_)
  # initialize for scope purposes
  discInputFake, foolDisc = 0f0, 0f0
  discTrueLog, discFalseLog, vfMAE, mse = 0f0, 0f0, 0f0, 0f0
  foolDiscMult, mseMult, vfMAEmult = 1.6f0, 25f0, 10f0
  function genLoss(genOutput) # generator loss. Defined here for scope purposes
    mse = (genOutput .- realTopology) .^ 2 |> mean # topology MSE
    l1 = abs.(genOutput .- realTopology) |> mean # topology l1 error
    # volume fraction mean absolute error
    # vfMAE = abs.(volFrac(genOutput) .- volFrac(realTopology)) |> mean
    # discriminator input with FAKE topology
    discInputFake = solidify(genInput, FEAinfo, genOutput) |> gpu
    # discriminator's output for FAKE topology
    discOutFake = discInputFake |> disc |> cpu |> reshapeDiscOut
    # println("foolDisc")
    foolDisc = logitBinCrossEntNoise(discOutFake, 0.85)
    # foolDisc = logitBinCrossEnt(discOutFake, 0.85)
    # generator's final loss
    # return logitBinCrossEnt(discOutFake, 1) + 10_000 * mse + 1 * absError
    # return foolDiscMult * foolDisc + mseMult * mse + vfMAEmult * vfMAE
    return foolDiscMult * foolDisc + mseMult * mse + 80 * l1
  end
  function discLoss(discOutReal, discOutFake) # discriminator loss
    # discTrueLog = logitBinCrossEnt(discOutReal, 0.85)
    # println("discTrueLog")
    discTrueLog = logitBinCrossEntNoise(discOutReal, 0.85)
    # discFalseLog = logitBinCrossEnt(discOutFake, 0.15)
    # println("discFalseLog")
    discFalseLog = logitBinCrossEntNoise(discOutFake, 0.15)
    return discTrueLog + discFalseLog
  end
  # discriminator input with REAL topology
  discInputReal = solidify(genInput, FEAinfo, realTopology)::Array{Float32, 4} |> gpu
  # get generator's loss value and gradient
  if goal == :train
    genLossVal, genGrads = withgradient(
      gen -> genLoss(gen(genInput |> gpu) |> cpu |> padGen), gen
    )
  else
    genLossVal = genLoss(gen(genInput |> gpu) |> cpu |> padGen)
  end
  if goal == :train
    # get discriminator's loss value and gradient
    discLossVal, discGrads = withgradient(
      disc -> discLoss(
          disc(discInputReal) |> cpu |> reshapeDiscOut,
          disc(discInputFake) |> cpu |> reshapeDiscOut
      ),
      disc
    )
  else
    discLossVal = discLoss(
        disc(discInputReal) |> cpu |> reshapeDiscOut,
        disc(discInputFake) |> cpu |> reshapeDiscOut
    )
  end
  # log histories
  logBatch && logBatchGenVals(metaData_, foolDiscMult * foolDisc, mseMult * mse, vfMAEmult * vfMAE)
  logBatch && logBatchDiscVals(metaData_, discTrueLog, discFalseLog)
  if goal == :train
    return genGrads, genLossVal, discGrads, discLossVal
  else
    return (0,), genLossVal, (0,), discLossVal
  end
end

# Loss for NN architectures with more than one output
function multiLoss(output, target; lossFun)
  # Compares each prediction/target pair from the
  # function getLoaders() with multi-output: x positions,
  # y positions, components of first load, components of second load
  return mean([lossFun(modelOut, truth) for (modelOut, truth) in zip(output, target)])
end

