import Pkg
Pkg.activate(".")
Pkg. instantiate()
include(projPath * "QTOutils.jl")
# julia --sysimage=C:/mySysImage.so topoGAN.jl
# GANfolderPath = projPath * "networks/GANplots/12-10.0%-0Ww6"
# plotGANValHist(0, 0, "aaa";
#   metaDataName = datasetPath * "data/checkpoints/12-17T12-08-12metaData.txt"
# )

const batchSize = 64
normalizeDataset::Bool = true # choose to normalize data in [-1; 1]
const startTime = timeNow()
# lineScale = identity # log10/identity
percentageDataset::Float64 = 0.1 # fraction of dataset to be used

@time expMetaData = trainGANs(;
  genOpt_ = Flux.Optimise.Adam(),
  discOpt_ = Flux.Optimise.Adam(),
  genName_ = "12-19T06-55-35-0gen.bson",
  discName_ = "12-19T06-55-52-0disc.bson",
  metaDataName = projPath * "networks/GANplots/26-10.0%-aUmr/12-19T06-59-11metaData.txt",
  originalFolder = "C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/networks/GANplots/26-10.0%-aUmr",
  epochs = 10,
  valFreq = 2
)
saveGANs(expMetaData, 0; finalSave = true) # save final models
GANreport(expMetaData) # create report