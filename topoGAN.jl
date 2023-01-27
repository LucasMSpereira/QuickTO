import Pkg
Pkg.activate(".")
Pkg. instantiate()
include(projPath * "QTOutils.jl")

const batchSize = 64
const normalizeDataset = false # choose to normalize data in [-1; 1]
const startTime = timeNow()
const percentageDataset = 0.4 # fraction of dataset to be used
const wasserstein = true

@time expMetaData = trainGANs(;
  genOpt_ = Flux.Optimise.AdamW(1e-4),
  discOpt_ = Flux.Optimiser(Flux.ClipNorm(1.0), Flux.Optimise.AdamW(0.01e-4)),
  # genName_ = "01-26T21-29-09-0gen.bson",
  # discName_ = "01-26T21-29-44-0disc.bson",
  # metaDataName = projPath * "networks/GANplots/01-26T17-52-39-9slP/01-26T19-51-33metaData.txt",
  # originalFolder = projPath * "networks/GANplots/01-26T17-52-39-9slP",
  epochs = 6,
  valFreq = 3,
  architectures = (
    # convNextModel(96, [3, 3, 9, 3], 0.5),
    # convNextModel(128, [3, 3, 27, 3], 0.5),
    convNextModel(192, [3, 3, 27, 3], 0.5),
    # U_SE_ResNetGenerator(),
    # patchGANdisc()
    topologyGANdisc(; drop = true)
  )
)
# saveGANs(expMetaData, 0; finalSave = true) # save final models
switchTraining(expMetaData, false)
GANreport(expMetaData) # create report