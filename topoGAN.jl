import Pkg
Pkg.activate(".")
Pkg. instantiate()
include(projPath * "QTOutils.jl")

const batchSize = 64
const normalizeDataset = true # choose to normalize data in [-1; 1]
const startTime = timeNow()
const percentageDataset = 0.11 # fraction of dataset to be used
const wasserstein = false

@time expMetaData = trainGANs(;
  genOpt_ = Flux.Optimise.Adam(),
  discOpt_ = Flux.Optimise.Adam(),
  # genName_ = "01-12T15-51-32-0gen.bson",
  # discName_ = "01-12T15-51-54-0disc.bson",
  # metaDataName = projPath * "networks/GANplots/30-18.0%-hFYb/01-12T15-53-45metaData.txt",
  # originalFolder = projPath * "networks/GANplots/30-18.0%-hFYb",
  epochs = 1,
  valFreq = 1,
  architectures = (
    # convNextModel(96, [3, 3, 9, 3], 0.5),
    convNextModel(192, [3, 3, 27, 3], 0.5),
    topologyGANdisc()
  )
)
# saveGANs(expMetaData, 0; finalSave = true) # save final models
switchTraining(expMetaData, false)
GANreport(expMetaData) # create report