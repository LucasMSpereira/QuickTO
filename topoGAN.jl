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
  # genName_ = "01-04T16-48-20-0gen.bson",
  # discName_ = "01-04T16-48-36-0disc.bson",
  # metaDataName = projPath * "networks/GANplots/12-18.0%-7W7B/01-04T16-50-57metaData.txt",
  # originalFolder = "C:/Users/kaoid/My Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/networks/GANplots/12-18.0%-7W7B",
  epochs = 15,
  valFreq = 3,
  architectures = (
    convNextModel(192, [3, 3, 27, 3], 0.5),
    topologyGANdisc()
  )
)
saveGANs(expMetaData, 0; finalSave = true) # save final models
GANreport(expMetaData) # create report
#= convnext sizes
:tiny => ([3, 3, 9, 3], [96, 192, 384, 768]),
:small => ([3, 3, 27, 3], [96, 192, 384, 768]),
:base => ([3, 3, 27, 3], [128, 256, 512, 1024]),
:large => ([3, 3, 27, 3], [192, 384, 768, 1536]),
:xlarge => ([3, 3, 27, 3], [256, 512, 1024, 2048]))
=#