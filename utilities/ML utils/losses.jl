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

# g_loss_final. Loss used to train generator
function generatorLoss(genLossLogit_, msError_, vfAE_)
  return genLossLogit_ + l1λ * msError_ + l2λ * vfAE_
end

# gan_loss_d. Loss used to train discriminator
discLoss(gLossDreal_, gLossDfake_) = gLossDreal_ + gLossDfake_

# use batch data and models to obtain gradients and losses
function GANgrads(gen, disc, genInput, FEAinfo, realTopology)
  # concatenate data to create discriminator input with REAL topology
  discInputReal = solidify(genInput, FEAinfo, realTopology) |> gpu
  # discriminator's output for input with REAL topology
  discOutReal = discInputReal |> disc |> cpu |> reshapeDiscOut
  # initialize for scope purposes
  discOutFake, genInputGPU, discInputFake = 0.0, 0.0, 0.0
  # function to calculate loss of generator. It's
  # defined here for scope purposes
  function genLoss(genOutput)
    # batch MSE of generator output (FAKE topologies)
    mse = (genOutput .- realTopology) .^ 2 |> mean
    # batch mean absolute error of generator output (FAKE topologies)
    absError = abs.(volFrac(genOutput) .- volFrac(realTopology)) |> mean
    # concatenate data to create discriminator input with FAKE topology
    discInputFake = solidify(genInput, FEAinfo, genOutput) |> gpu
    # discriminator's output for input with FAKE topology
    discOutFake = discInputFake |> disc |> cpu |> reshapeDiscOut
    # batch mean binary cross-entropy loss for generator
    genCE = logitBinCrossEnt(discOutFake, 1)
    return genCE + 10_000 * mse + 1 * absError # final generator loss
  end
  # function to calculate loss of discriminator. It's
  # defined here for scope purposes
  discLoss(discOutReal) = logitBinCrossEnt(discOutReal, 1) + logitBinCrossEnt(discOutFake, 0)
  # gradients and final losses of both NNs
  genInputGPU = genInput |> gpu
  genLossVal, genGrads = withgradient(() -> genLoss(genInputGPU |> gen |> cpu), Flux.params(gen))
  discLossVal, discGrads = withgradient(() -> discLoss(discOutReal), Flux.params(disc))
  return genGrads, genLossVal, discGrads, discLossVal
end

# Loss for NN architectures with more than one output
function multiLoss(output, target; lossFun)
  # Compares each prediction/target pair from the
  # function getLoaders() with multi-output: x positions,
  # y positions, components of first load, components of second load
  return mean([lossFun(modelOut, truth) for (modelOut, truth) in zip(output, target)])
end

# Errors used for loss calculation in topologyGAN
function topoGANerrors(fakeTopo_, realTopo_)
  # fakeTopo_: batch of predicted (FAKE) topologies
  # realTopo_: batch of real topologies
  return (
    # mean of batch volume fraction absolute error
    abs.(volFrac(fakeTopo_) .- volFrac(realTopo_)) |> mean,
    # batch mean squared topology error
    (fakeTopo_ .- realTopo_) .^ 2 |> mean
    )
end


# logits representing the discriminator's behavior
function topoGANlogits(logitFake_, logitReal_)
  discReal = ones(Float32, size(logitFake_))
  discFake = zeros(Float32, size(discReal))
  return (
    # generator wants discriminator to output 1 to FAKE topologies
    Flux.Losses.logitbinarycrossentropy(logitFake_, discReal), # gan_loss_g
    # discriminator should output 1 to TRUE topologies
    Flux.Losses.logitbinarycrossentropy(logitReal_, discReal), # gan_loss_d_real
    # discriminator should output 0 to FAKE topologies
    Flux.Losses.logitbinarycrossentropy(logitFake_, discFake), # gan_loss_d_fake
  )
end