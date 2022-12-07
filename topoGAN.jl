include("./QTOutils.jl")
const batchSize = 64
# binaries for logit binary cross-entropy
const discBinaryReal = ones(Float32, batchSize)
const discBinaryFake = zeros(Float32, batchSize)
percentageDataset::Float64 = 0.25
Random.seed!(3111)

function trainGANs(; opt = Flux.Optimise.Adam(), genName_ = " ", discName_ = " ", metaDataPath = "")
  # object with metadata. includes instantiation of NNs,
  # optimiser, dataloaders, training configurations,
  # validation histories, and test losses
  metaData = GANmetaData(
    if genName_ == " " # new NNs
      (U_SE_ResNetGenerator(), topologyGANdisc())
    else # use input path to load previous models
      loadGANs(genName_, discName_)
    end...,
    opt, epochTrainConfig(12, 4),
    # datasetPath * "data/checkpoints/2022-12-06T15-13-19metaData.txt"
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
[[GC.gc() CUDA.reclaim()] for _ in 1:2]
experimentMetaData = trainGANs(;
  opt = Flux.Optimise.NAdam(1e-4),
)
saveGANs(experimentMetaData; finalSave = true) # save final models
GANreport(
  string(experimentMetaData.trainConfig.epochs) * "-" * string(round(Int, percentageDataset * 100)) *
  "%-" * string(experimentMetaData.trainConfig.validFreq) * "-" * sciNotation(lr, 1),
  experimentMetaData
)