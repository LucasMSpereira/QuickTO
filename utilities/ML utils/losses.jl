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

# use batch data and models to obtain gradients and losses
function GANgrads(gen, disc, genInput, FEAinfo, realTopology)
  discOutFake, discInputFake = 0.0, 0.0 # initialize for scope purposes
  function genLoss(genOutput) # generator loss. Defined here for scope purposes
    mse = (genOutput .- realTopology) .^ 2 |> mean # topology MSE
    # volume fraction MAE
    absError = abs.(volFrac(genOutput) .- volFrac(realTopology)) |> mean
    # discriminator input with FAKE topology
    discInputFake = solidify(genInput, FEAinfo, genOutput) |> gpu
    # discriminator's output for FAKE topology
    discOutFake = discInputFake |> disc |> cpu |> reshapeDiscOut
    # generator's final loss
    return logitBinCrossEnt(discOutFake, 1)  + 10_000 * mse + 1 * absError
  end
  function discLoss(discOutReal, discOutFake) # discriminator loss
    return logitBinCrossEnt(discOutReal, 1) + logitBinCrossEnt(discOutFake, 0)
  end
  genInputGPU = genInput |> gpu # copy genertor's input to GPU
  # discriminator input with REAL topology
  discInputReal = solidify(genInput, FEAinfo, realTopology) |> gpu
  # get generator's loss value and gradient
  genLossVal_, genGrads_ = withgradient(
    gen -> genLoss(gen(genInputGPU) |> cpu), gen
  )
  # get discriminator's loss value and gradient
  discLossVal_, discGrads_ = withgradient(
    disc -> discLoss(
        disc(discInputReal) |> cpu |> reshapeDiscOut,
        disc(discInputFake) |> cpu |> reshapeDiscOut
    ),
    disc
  )
  return genGrads_, genLossVal_, discGrads_, discLossVal_
end

# Loss for NN architectures with more than one output
function multiLoss(output, target; lossFun)
  # Compares each prediction/target pair from the
  # function getLoaders() with multi-output: x positions,
  # y positions, components of first load, components of second load
  return mean([lossFun(modelOut, truth) for (modelOut, truth) in zip(output, target)])
end