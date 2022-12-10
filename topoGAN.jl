# import Pkg
# Pkg.activate(".")
# Pkg. instantiate()
include("./QTOutils.jl")
# julia --sysimage=C:/mySysImage.so topoGAN.jl
plotGANValHist(
  0, 0, "./networks/GANplots/14-3.0%-2-2022-12-09T21-53-28",
  "2022-12-09T21-53-25"; metaDataPath = "2022-12-09T21-53-25metaData.txt"
)
const batchSize = 64
# binaries for logit binary cross-entropy
const discBinaryReal = ones(Float32, batchSize)
const discBinaryFake = zeros(Float32, batchSize)
percentageDataset::Float64 = 0.03 # fraction of dataset to be used
normalizeDataset::Bool = true # choose to normalize data in [-1; 1]
lineScale = identity # log10/identity
Random.seed!(3111)

function trainGANs(;
  genOpt_, discOpt_, genName_ = " ", discName_ = " ", metaDataName = ""
)
  # object with metadata. includes instantiation of NNs,
  # optimisers, dataloaders, training configurations,
  # validation histories, and test losses
  if genName_ == " " # new NNs
    metaData = GANmetaData(
    U_SE_ResNetGenerator(), topologyGANdisc(),
    genOpt_, discOpt_, epochTrainConfig(14, 2)
    )
  else # use input path to load previous models
    metaData = GANmetaData(
      loadGANs(genName_, discName_)...,
      genOpt_, discOpt_, epochTrainConfig(14, 2),
      metaDataName
    )
  end
  println("Starting training ", timeNow())
  if typeof(metaData.trainConfig) == earlyStopTrainConfig
    @suppress_err earlyStopGANs(metaData) # train with early-stopping
  elseif typeof(metaData.trainConfig) == epochTrainConfig
    @suppress_err fixedEpochGANs(metaData) # train for fixed number of epochs
  end
  # println("Testing ", timeNow())
  switchTraining(metaData, false) # disable model updating during test
  # metaData(GANepoch!(metaData, :test); context = :test) # test GANs
  metaData((0.0, 0.0); context = :test)
  switchTraining(metaData, true) # reenable model updating
  return metaData
end
[[GC.gc() CUDA.reclaim()] for _ in 1:2]
@time expMetaData = trainGANs(;
  genOpt_ = Flux.Optimise.Adam(4e-3),
  discOpt_ = Flux.Optimise.Adam(4e-3),
  genName_ = "2022-12-09T17-40-41gen.bson",
  discName_ = "2022-12-09T17-40-59disc.bson",
  metaDataName = "2022-12-09T17-41-03metaData.txt"
)
saveGANs(expMetaData; finalSave = true) # save final models
GANreport(expMetaData)