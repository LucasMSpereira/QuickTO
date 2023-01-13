import Pkg
Pkg.activate(".")
Pkg. instantiate()
include(projPath * "QTOutils.jl")
# julia --sysimage=C:/mySysImage.so topoGAN.jl

const batchSize = 64
const normalizeDataset = true # choose to normalize data in [-1; 1]
const startTime = timeNow()
# const to = TimerOutput()
const percentageDataset = 0.18 # fraction of dataset to be used
# LinearAlgebra.norm(::Nothing, p::Real=2) = false

@time expMetaData = trainGANs(;
  genOpt_ = Flux.Optimise.Adam(),
  discOpt_ = Flux.Optimise.Adam(),
  genName_ = "01-12T15-51-32-0gen.bson",
  discName_ = "01-12T15-51-54-0disc.bson",
  metaDataName = projPath * "networks/GANplots/30-18.0%-hFYb/01-12T15-53-45metaData.txt",
  originalFolder = projPath * "networks/GANplots/30-18.0%-hFYb",
  epochs = 30,
  valFreq = 3,
  # architectures = (
  #   convNextModel(192, [3, 3, 27, 3], 0.5),
  #   topologyGANdisc()
  # )
)
saveGANs(expMetaData, 0; finalSave = true) # save final models
switchTraining(expMetaData, false) # disable model updating during validation
GANreport(expMetaData) # create report