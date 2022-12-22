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
complianceLoss::Bool = true # include compliance in loss
percentageDataset::Float64 = 0.1 # fraction of dataset to be used
# LinearAlgebra.norm(::Nothing, p::Real=2) = false

@time expMetaData = trainGANs(;
  genOpt_ = Flux.Optimise.Adam(),
  discOpt_ = Flux.Optimise.Adam(),
  # genName_ = "12-19T14-46-29-0gen.bson",
  # discName_ = "12-19T14-46-51-0disc.bson",
  # metaDataName = projPath * "networks/GANplots/26-10.0%-aUmr/12-19T14-50-10metaData.txt",
  # originalFolder = "C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/networks/GANplots/26-10.0%-aUmr",
  epochs = 27,
  valFreq = 3
)
saveGANs(expMetaData, 0; finalSave = true) # save final models
GANreport(expMetaData) # create report
save_object(datasetPath * "data/checkpoints/" * timeNow() * "metadata.jld2", expMetaData)