include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
const batchSize = 64
# binaries for logit binary cross-entropy
const discBinaryReal = ones(Float32, batchSize)
const discBinaryFake = zeros(Float32, batchSize)
percentageDataset::Float64 = 0.05
Random.seed!(3111)

# begin
  #     m = GANmetaData(
  #       Chain(Conv((3, 3), 1 => 1)), Chain(Conv((3, 3), 1 => 1)),
  #       Flux.Optimise.Adam(), epochTrainConfig(67, 5)
  #     )
  #     [m((randBetween(100, 1000), randBetween(0, 5))) for _ in 1:15]
  #     m((randBetween(100, 1000), randBetween(0, 5)); context = :test)
  #     # GANreport("modelName", m)
  #     writeLosses(m)
# end
function GANgradsTest(gen, disc, genInput, FEAinfo, realTopology)
  # concatenate data to create discriminator input with REAL topology
  discInputReal = solidify(genInput, FEAinfo, realTopology) |> gpu
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
  # discLoss(discOutReal) = logitBinCrossEnt(discOutReal, 1) + logitBinCrossEnt(discOutFake, 0)
  discLoss(discOutReal) = logitBinCrossEnt(discOutFake, 0)
  # gradients and final losses of both NNs
  genInputGPU = genInput |> gpu
  genLossVal, genGrads = withgradient(() -> genLoss(genInputGPU |> gen |> cpu), Flux.params(gen))
  discLossVal, discGrads = withgradient(() -> discLoss(discInputReal |> disc |> cpu |> reshapeDiscOut), Flux.params(disc))
  return genGrads, genLossVal, discGrads, discLossVal
end
function testGANepoch!(metaData, goal)
    !in(goal, [:train, :validate, :test]) && error("GANepoch!() called with invalid 'goal'.")
    genGrads, genLossVal, discGrads, discLossVal = 0, 0, 0, 0
    # initialize variables related to whole epoch
    genLossHist, discLossHist, batchCount = 0.0, 0.0, 0
    groupFiles = defineGroupFiles(metaData, goal)
    # loop in groups of files used for current split
    for group in groupFiles
      # get loader with data for current group
      currentLoader = GANdataLoader(metaData, goal, group)
      # each batch of current epoch
      @suppress_err for (genInput, FEAinfo, realTopology) in currentLoader
        batchCount += 1
        GC.gc(); CUDA.reclaim() # avoid GPU memory issues
        # use NNs, and get gradients and losses for current batch
        genGrads, genLossVal, discGrads, discLossVal = GANgradsTest(
          metaData.generator, metaData.discriminator, genInput, FEAinfo, realTopology
        )
        if goal == :train # update NNs parameters in case of training
          Flux.Optimise.update!(metaData.opt, Flux.params(metaData.generator), genGrads)
          Flux.Optimise.update!(metaData.opt, Flux.params(metaData.discriminator), discGrads)
        end
        # acumulate batch losses
        genLossHist += genLossVal; discLossHist += discLossVal
        break
      end
    end
    # return avg losses for current epoch
    return genGrads, genLossVal, discGrads, discLossVal
end
m = GANmetaData(U_SE_ResNetGenerator(), topologyGANdisc(), Flux.Optimise.Adam(), epochTrainConfig(5, 5))
@time gGrads, gLoss, dGrads, dLoss = testGANepoch!(m, :test)
[[GC.gc() CUDA.reclaim()] for _ in 1:2]
for (dictionary, name) in zip([gGrads.grads, dGrads.grads], ["generator\n", "discriminator\n"])
    nothingness = 0
    println(name, length(dictionary))
    for (key, value) in dictionary
        if value === nothing
            nothingness += 1
            println(rpad(nothingness, 5))
        end
    end
    println()
end
gParams = Flux.params(m.generator |> cpu);
dParams = Flux.params(m.discriminator |> cpu);
@show norm(gGrads); @show norm(dGrads);