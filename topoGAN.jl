import Pkg
Pkg.activate(".")
Pkg. instantiate()
include("./QTOutils.jl")
# julia --sysimage=C:/mySysImage.so topoGAN.jl
# plotGANValHist(
#   0, 0, "./networks/GANplots/66-10%-4-2022-12-11T15-04-41",
#   "12-11T15-04-41"; metaDataName = "12-11T15-04-41metaData.txt"
# )

function trainGANs(;
  genOpt_, discOpt_, genName_ = " ", discName_ = " ", metaDataName = "",
  epochs, valFreq
)
  # object with metadata. includes instantiation of NNs,
  # optimisers, dataloaders, training configurations,
  # validation histories, and test losses
  if genName_ == " " # new NNs
    metaData = GANmetaData(
    U_SE_ResNetGenerator(), topologyGANdisc(),
    genOpt_, discOpt_, epochTrainConfig(epochs, valFreq)
    )
  else # use input path to load previous models
    metaData = GANmetaData(
      loadGANs(genName_, discName_)...,
      genOpt_, discOpt_, epochTrainConfig(epochs, valFreq),
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

const batchSize = 64
# binaries for logit binary cross-entropy
const discBinaryReal = ones(Float32, batchSize)
const discBinaryFake = zeros(Float32, batchSize)
normalizeDataset::Bool = true # choose to normalize data in [-1; 1]
lineScale = identity # log10/identity
Random.seed!(3111)
percentageDataset::Float64 = 0.1 # fraction of dataset to be used

const runningInColab = 1

@time expMetaData = trainGANs(;
  genOpt_ = Flux.Optimise.Adam(4e-3),
  discOpt_ = Flux.Optimise.Adam(4e-3),
  genName_ = "12-10T23-05-24-9gen.bson",
  discName_ = "12-10T23-05-48-9disc.bson",
  metaDataName = "2022-12-10T23-05-50metaData.txt",
  epochs = 7,
  valFreq = 2
)
saveGANs(expMetaData, 0; finalSave = true) # save final models
GANreport(expMetaData)