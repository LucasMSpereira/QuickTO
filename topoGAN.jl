import Pkg
Pkg.activate(".")
Pkg. instantiate()
include(projPath * "QTOutils.jl")

const batchSize = 64
const normalizeDataset = true # choose to normalize data in [-1; 1]
const startTime = timeNow()
const percentageDataset = 0.05 # fraction of dataset to be used
const wasserstein = true

@time expMetaData = trainGANs(;
  genOpt_ = Flux.Optimise.OAdam(),
  discOpt_ = Flux.Optimiser(Flux.ClipNorm(1.0), Flux.Optimise.OAdam()),
  # genName_ = "01-20T14-20-59-0gen.bson",
  # discName_ = "01-20T14-21-19-0disc.bson",
  # metaDataName = projPath * "networks/GANplots/01-20T11-55-39-e5Lw/01-20T13-15-09metaData.txt",
  # originalFolder = projPath * "networks/GANplots/01-20T11-55-39-e5Lw",
  epochs = 2,
  valFreq = 1,
  architectures = (
    # convNextModel(96, [3, 3, 9, 3], 0.5),
    # convNextModel(128, [3, 3, 27, 3], 0.5),
    convNextModel(192, [3, 3, 27, 3], 0.5),
    # U_SE_ResNetGenerator(),
    patchGANdisc()
    # topologyGANdisc()
  )
)
# saveGANs(expMetaData, 0; finalSave = true) # save final models
switchTraining(expMetaData, false)
GANreport(expMetaData) # create report