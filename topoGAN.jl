import Pkg
Pkg.activate(".")
Pkg. instantiate()
include(projPath * "QTOutils.jl")
# julia --sysimage=C:/mySysImage.so topoGAN.jl
# global GANfolderPath = createGANfolder(GANmetaData(
#   Chain(Conv((3, 3), 1 => 1)), Chain(Conv((3, 3), 1 => 1)),
#   Flux.Optimise.Adam(),Flux.Optimise.Adam(), epochTrainConfig(67, 5),
#   "12-14T20-23-51metaData.txt"
# ))::String
# plotGANValHist(
#   0, 0, "./networks/GANplots/67-10.0%-5-12-14T23-19-14",
#   "aaa"; metaDataName = "12-14T20-23-51metaData.txt"
# )

function trainGANs(;
  genOpt_, discOpt_, genName_ = " ", discName_ = " ",
  metaDataName = "", originalFolder = " ", epochs, valFreq
)
  # object with metadata. includes instantiation of NNs,
  # optimisers, dataloaders, training configurations,
  # validation histories, and test losses
  if genName_ == " " # new NNs
    metaData = GANmetaData(
      U_SE_ResNetGenerator(), topologyGANdisc(),
      genOpt_, discOpt_, epochTrainConfig(epochs, valFreq)
    )
    # create folder to store plots and report
    global GANfolderPath = createGANfolder(metaData)::String
  else # use input path to load previous models
    # create folder to store plots and report
    global GANfolderPath = originalFolder
    metaData = GANmetaData(
      loadGANs(genName_, discName_)...,
      genOpt_, discOpt_, epochTrainConfig(epochs, valFreq),
      metaDataName
    )
  end
  println("Starting training ", timeNow())
  if typeof(metaData.trainConfig) == earlyStopTrainConfig
    @suppress_err earlyStopGANs(metaData) # train with early-stopping
  elseif typeof(metaData.trainConfig) == epochTrainConfig
    @suppress_err fixedEpochGANs(metaData) # train for fixed number of epochs
  end
  println("Testing ", timeNow())
  switchTraining(metaData, false) # disable model updating during test
  metaData(GANepoch!(metaData, :test); context = :test) # test GANs
  # metaData((0.0, 0.0); context = :test)
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
  genOpt_ = Flux.Optimise.Adam(2.5e-3),
  discOpt_ = Flux.Optimise.Adam(2.5e-3),
  # genName_ = "12-15T17-02-21-0gen.bson",
  # discName_ = "12-15T17-02-38-0disc.bson",
  # metaDataName = projPath * "networks/GANplots/26-10.0%-2-12-15T08-25-51/12-15T17-06-46metaData.txt",
  epochs = 12,
  valFreq = 2
)
saveGANs(expMetaData, 0; finalSave = true) # save final models
GANreport(expMetaData) # create report