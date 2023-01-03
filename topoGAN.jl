import Pkg
Pkg.activate(".")
Pkg. instantiate()
include(projPath * "QTOutils.jl")
# julia --sysimage=C:/mySysImage.so topoGAN.jl
# GANfolderPath = projPath * "networks/GANplots/12-10.0%-aaaa"
# plotGANValHist(0, 0, "aaa";
#   metaDataName = datasetPath * "data/checkpoints/12-20T12-42-26metaData.txt"
# )

const batchSize = 64
normalizeDataset::Bool = true # choose to normalize data in [-1; 1]
const startTime = timeNow()
# const to = TimerOutput()
percentageDataset::Float64 = 0.18 # fraction of dataset to be used
# LinearAlgebra.norm(::Nothing, p::Real=2) = false

@time expMetaData = trainGANs(;
  genOpt_ = Flux.Optimise.Adam(2e-3),
  discOpt_ = Flux.Optimise.Adam(2e-3),
  genName_ = "01-02T17-06-36-0gen.bson",
  discName_ = "01-02T17-06-54-0disc.bson",
  metaDataName = projPath * "networks/GANplots/12-18.0%-7W7B/01-02T17-09-08metaData.txt",
  originalFolder = "C:/Users/kaoid/My Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/networks/GANplots/12-18.0%-7W7B",
  epochs = 15,
  valFreq = 3
)
saveGANs(expMetaData, 0; finalSave = true) # save final models
GANreport(expMetaData) # create report
save_object(datasetPath * "data/checkpoints/" * timeNow() * "metadata.jld2", expMetaData)

plotGANlogs(datasetPath * "data/checkpoints/01-02T17-09-09metadata.jld2")