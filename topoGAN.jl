import Pkg
Pkg.activate(".")
Pkg. instantiate()
include(projPath * "QTOutils.jl")
# julia --sysimage=C:/mySysImage.so topoGAN.jl
# plotGANValHist(
#   0, 0, "./networks/GANplots/66-10%-4-2022-12-11T15-04-41",
#   "12-11T15-04-41"; metaDataName = "12-11T15-04-41metaData.txt"
# )

function trainGANs(;
  genOpt_, discOpt_, genName_ = " ", discName_ = " ", metaDataName = "",
  epochs, valFreq
)
  # object with metadata. includes instantiation of NNs,
  # optimisers, dataloaders, training configurations,
  # validation histories, and test losses
  if genName_ == " " # new NNs
    metaData = GANmetaData(
    U_SE_ResNetGenerator(), topologyGANdisc(),
    genOpt_, discOpt_, epochTrainConfig(epochs, valFreq)
    )
  else # use input path to load previous models
    metaData = GANmetaData(
      loadGANs(genName_, discName_)...,
      genOpt_, discOpt_, epochTrainConfig(epochs, valFreq),
      metaDataName
    )
  end
  # create folder to store plots and report
  global GANfolderPath = createGANfolder(metaData)::String
  println("Starting training ", timeNow())
  if typeof(metaData.trainConfig) == earlyStopTrainConfig
    @suppress_err earlyStopGANs(metaData) # train with early-stopping
  elseif typeof(metaData.trainConfig) == epochTrainConfig
    @suppress_err fixedEpochGANs(metaData) # train for fixed number of epochs
  end
  # println("Testing ", timeNow())
  switchTraining(metaData, false) # disable model updating during test
  # metaData(GANepoch!(metaData, :test); context = :test) # test GANs
  metaData((0.0, 0.0); context = :test)
  switchTraining(metaData, true) # reenable model updating
  return metaData
end

const batchSize = 64
normalizeDataset::Bool = true # choose to normalize data in [-1; 1]
const startTime = timeNow()
# lineScale = identity # log10/identity
Random.seed!(3111)
percentageDataset::Float64 = 0.1 # fraction of dataset to be used

@time expMetaData = trainGANs(;
  genOpt_ = Flux.Optimise.Adam(),
  discOpt_ = Flux.Optimise.Adam(),
  epochs = 12,
  valFreq = 2
)
saveGANs(expMetaData, 0; finalSave = true) # save final models
GANreport(expMetaData) # create report