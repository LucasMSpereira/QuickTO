include("./utilsColab.jl")
const batchSize = 64
# binaries for logit binary cross-entropy
const discBinaryReal = ones(Float32, batchSize)
const discBinaryFake = zeros(Float32, batchSize)
percentageDataset::Float64 = 0.05
Random.seed!(3111)

function trainGANs(;
  opt = Flux.Optimise.Adam(), genPath_ = " ", discPath_ = " ",
  epochs, valFreq
)
  # object with metadata. includes instantiation of NNs,
  # optimiser, dataloaders, training configurations,
  # validation histories, and test losses
  metaData = GANmetaData(
    if genPath_ == " " # new NNs
      (U_SE_ResNetGenerator(), topologyGANdisc())
    else # use input path to load previous models
      loadGANs(genPath_, discPath_)
    end...,
    opt, epochTrainConfig(epochs, valFreq)
  )
  println("Starting training ", timeNow())
  if typeof(metaData.trainConfig) == earlyStopTrainConfig
    @suppress_err earlyStopGANs(metaData) # train with early-stopping
  elseif typeof(metaData.trainConfig) == epochTrainConfig
    @suppress_err fixedEpochGANs(metaData) # train for fixed number of epochs
  end
  println("Testing ", timeNow())
  switchTraining(metaData, false) # disable model updating during test
  metaData(GANepoch!(metaData, :test); context = :test) # test GANs
  switchTraining(metaData, true) # reenable model updating
  return metaData
end
experimentMetaData = trainGANs(; opt = Flux.Optimise.NAdam(lr));
saveGANs(experimentMetaData; finalSave = true) # save final models
GANreport(
  string(experimentMetaData.trainConfig.epochs) * "-" * string(round(Int, percentageDataset * 100)) *
  "%-" * string(experimentMetaData.trainConfig.validFreq) * "-" * sciNotation(lr, 1),
  experimentMetaData
)