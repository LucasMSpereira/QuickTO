import Pkg
Pkg.activate(".")
Pkg. instantiate()
include(projPath * "QTOutils.jl")

const batchSize = 64
const normalizeDataset = false # choose to normalize data in [-1; 1]
const startTime = timeNow()
const percentageDataset = 0.6 # fraction of dataset to be used
const wasserstein = true

@time expMetaData = trainGANs(;
  genOpt_ = Flux.Optimise.AdamW(1e-4),
  discOpt_ = Flux.Optimiser(Flux.ClipNorm(1.0), Flux.Optimise.AdamW(1e-4)),
  # genName_ = "01-30T13-49-30-3gen.bson",
  # discName_ = "01-30T13-50-04-3disc.bson",
  # metaDataName = projPath * "networks/GANplots/01-29T09-45-03-Bvp4/01-29T20-07-39metaData.txt",
  # originalFolder = projPath * "networks/GANplots/01-29T09-45-03-Bvp4/",
  architectures = (
    # convNextModel(96, [3, 3, 9, 3], 0.5),
    # convNextModel(128, [3, 3, 27, 3], 0.5),
    convNextModel(192, [3, 3, 27, 3], 0.5),
    # U_SE_ResNetGenerator(),
    # patchGANdisc()
    topologyGANdisc(; drop = 0.3)
  ),
  trainConfig = epochTrainConfig(3, 3)
  # trainConfig = earlyStopTrainConfig(
  #   1; earlyStopQuant = 2, earlyStopPercent = 5
  # )
)
saveGANs(expMetaData, 0; finalSave = true) # save final models
switchTraining(expMetaData, false)
GANreport(expMetaData) # create report