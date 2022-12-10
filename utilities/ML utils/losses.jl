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
function GANgrads(gen, disc, genInput, FEAinfo, realTopology)
  # initialize for scope purposes
  discOutFake, discInputFake = 0.0, 0.0
  function genLoss(genOutput) # generator loss. Defined here for scope purposes
    mse = (genOutput .- realTopology) .^ 2 |> mean # topology MSE
    absError = abs.(volFrac(genOutput) .- volFrac(realTopology)) |> mean # volume fraction MAE
    # discriminator input with FAKE topology
    discInputFake = solidify(genInput, FEAinfo, genOutput) |> gpu
    # discriminator's output for FAKE topology
    discOutFake = discInputFake |> disc |> cpu |> reshapeDiscOut
    # generator's final loss
    # return logitBinCrossEnt(discOutFake, 1) + 10_000 * mse + 1 * absError
    return logitBinCrossEnt(discOutFake, 1) + 10 * mse + 1 * absError
  end
  function discLoss(discOutReal, discOutFake) # discriminator loss
    # return logitBinCrossEnt(discOutReal, 1) + logitBinCrossEnt(discOutFake, 0)
    return logitBinCrossEnt(discOutReal, 0.85) + logitBinCrossEnt(discOutFake, 0)
  end
  genInputGPU = genInput |> gpu # copy genertor's input to GPU
  # discriminator input with REAL topology
  discInputReal = solidify(genInput, FEAinfo, realTopology) |> gpu
  # get generator's loss value and gradient
  genLossVal, genGrads = withgradient(
    gen -> genLoss(gen(genInputGPU) |> cpu |> padGen), gen
  )
  # get discriminator's loss value and gradient
  discLossVal, discGrads = withgradient(
    disc -> discLoss(
        disc(discInputReal) |> cpu |> reshapeDiscOut,
        disc(discInputFake) |> cpu |> reshapeDiscOut
    ),
    disc
  )
  return genGrads, genLossVal, discGrads, discLossVal
end

# Loss for NN architectures with more than one output
function multiLoss(output, target; lossFun)
  # Compares each prediction/target pair from the
  # function getLoaders() with multi-output: x positions,
  # y positions, components of first load, components of second load
  return mean([lossFun(modelOut, truth) for (modelOut, truth) in zip(output, target)])
end