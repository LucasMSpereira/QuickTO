include("./utilsColab.jl")
const batchSize = 64
# binaries for logit binary cross-entropy
const discBinaryReal = ones(Float32, batchSize)
const discBinaryFake = zeros(Float32, batchSize)
normalizeDataset::Bool = true # choose to normalize data in [-1; 1]
lineScale = log10 # log10/identity
Random.seed!(3111)

function trainGANs(;
  genOpt_, discOpt_, genName_ = " ", discName_ = " ", metaDataPath = "",
  epochs, valFreq
)
  # object with metadata. includes instantiation of NNs,
  # optimiser, dataloaders, training configurations,
  # validation histories, and test losses
  metaData = GANmetaData(
    if genName_ == " " # new NNs
      (U_SE_ResNetGenerator(), topologyGANdisc())
    else # use input path to load previous models
      loadGANs(genName_, discName_)
    end...,
    genOpt_, discOpt_, epochTrainConfig(epochs, valFreq),
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