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
  logBatch = rand() < 0.2 # choose if this batch will be logged
  gen = getGen(metaData_); disc = getDisc(metaData_)
  # initialize for scope purposes
  discInputFake, foolDisc = 0f0, 0f0
  discTrueLog, discFalseLog, vfMAE, mse = 0f0, 0f0, 0f0, 0f0
  foolDiscMult, mseMult, vfMAEmult = 1.6f0, 10f0, 10f0
  function genLoss(genOutput) # generator loss. Defined here for scope purposes
    mse = (genOutput .- realTopology) .^ 2 |> mean # topology MSE
    # l1 = abs.(genOutput .- realTopology) |> mean # topology l1 error
    # volume fraction mean absolute error
    # vfMAE = abs.(volFrac(genOutput) .- volFrac(realTopology)) |> mean
    # discriminator input with FAKE topology
    discInputFake = solidify(genInput, FEAinfo, genOutput) |> gpu
    # discriminator's output for FAKE topology
    discOutFake = discInputFake |> disc |> cpu
    foolDisc = logitBinCrossEntNoise(discOutFake, 0.85)
    # foolDisc = logitBinCrossEnt(discOutFake, 0.85)
    # generator's final loss
    # return logitBinCrossEnt(discOutFake, 1) + 10_000 * mse + 1 * absError
    # return foolDiscMult * foolDisc + mseMult * mse + vfMAEmult * vfMAE
    return foolDiscMult * foolDisc + mseMult * mse
  end
  function discLoss(discOutReal, discOutFake) # discriminator loss
    discTrueLog = logitBinCrossEntNoise(discOutReal, 0.85)
    discFalseLog = logitBinCrossEntNoise(discOutFake, 0.15)
    return 0.1 * (discTrueLog + discFalseLog)
  end
  # discriminator input with REAL topology
  discInputReal = solidify(genInput, FEAinfo, realTopology)::Array{Float32, 4} |> gpu
  if goal == :train
    genLossVal, genGrads = withgradient( # get generator's loss value and gradient
      gen -> genLoss(gen(genInput |> gpu) |> cpu |> padGen), gen
    )
  else
    genLossVal = genLoss(gen(genInput |> gpu) |> cpu |> padGen)
  end
  if goal == :train
    # get discriminator's loss value and gradient
    discLossVal, discGrads = withgradient(
      disc -> discLoss(
          disc(discInputReal) |> cpu,
          disc(discInputFake) |> cpu
      ),
      disc
    )
  else
    discLossVal = discLoss(
        disc(discInputReal) |> cpu,
        disc(discInputFake) |> cpu
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

function WGANgrads(
  metaData_::GANmetaData, goal::Symbol, genInput::Array{Float32},
  FEAinfo::Array{Float32}, realTopology::Array{Float32}
)::Tuple{Tuple{Any}, Float32, Tuple{Any}, Float32}
  logBatch = rand() < 0.4 # choose if this batch will be logged
  gen = getGen(metaData_); disc = getDisc(metaData_)
  genLossVal, discLossVal, genDoutFake = 0f0, 0f0, 0f0
  mse, discReal, discFake = 0f0, 0f0, 0f0
  genDoutFakeMult, mseMult = 40f0, 4000f0
  fakeTopology = zeros(Float32, (51, 141, 1, size(genInput, 4)))
  function genLoss(gen)
    fakeTopology = gen(genInput |> gpu) |> cpu |> padGen
    genDoutFake = solidify(
      genInput, FEAinfo, fakeTopology
    ) |> gpu |> disc |> cpu |> mean |> smoothSqrt
    mse = (fakeTopology .- realTopology) .^ 2 |> mean
    return -genDoutFakeMult * genDoutFake + mseMult * mse
  end
  function wganGPloss(discOutReal, discOutFake)
    discReal = smoothSqrt(mean(discOutReal)); discFake = smoothSqrt(mean(discOutFake))
    return discFake - discReal
  end
  if goal == :train
    genLossVal, genGrads = withgradient(gen -> genLoss(gen), gen)
    # discriminator inputs with real and fake topologies
    discInputReal = solidify(genInput, FEAinfo, realTopology) |> gpu
    discInputFake = solidify(genInput, FEAinfo, fakeTopology) |> gpu
    discLossVal, discGrads = withgradient(
      disc -> wganGPloss(
        disc(discInputReal) |> cpu, disc(discInputFake) |> cpu),
        disc
    )
  else
    fakeTopology = gen(genInput |> gpu) |> cpu |> padGen
    discInputReal = solidify(genInput, FEAinfo, realTopology) |> gpu
    discInputFake = solidify(genInput, FEAinfo, fakeTopology) |> gpu
    discLossVal = wganGPloss(
      disc(discInputReal) |> cpu, disc(discInputFake) |> cpu
    )
    genLossVal = genLoss(gen)
  end
  if logBatch
    logWGANloss(
      metaData_, discFake, discReal,
      -genDoutFakeMult * genDoutFake, mseMult * mse
    )
  end
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

