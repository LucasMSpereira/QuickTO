include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
const batchSize = 64
# binaries for logit binary cross-entropy
const discBinaryReal = ones(Float32, batchSize)
const discBinaryFake = zeros(Float32, batchSize)
percentageDataset::Float64 = 0.2
Random.seed!(3111)

function trainGANs(; opt = Flux.Optimise.Adam(), checkpoint = true)
  # object with metadata. includes instantiation of NNs,
  # optimiser, dataloaders, training configurations,
  # validation histories, and test losses
  metaData = GANmetaData(
    (checkpoint == true ? loadGANs() : (U_SE_ResNetGenerator(), topologyGANdisc()))...,
    opt, epochTrainConfig(8, 4)
  )
  println("Starting training: ", timeNow())
  if typeof(metaData.trainConfig) == earlyStopTrainConfig
    @suppress_err earlyStopGANs(metaData) # train with early-stopping
  elseif typeof(metaData.trainConfig) == epochTrainConfig
    @suppress_err fixedEpochGANs(metaData) # train for fixed number of epochs
  end
  switchTraining(metaData, false) # disable model updating during test
  metaData(GANepoch!(metaData, :test); context = :test) # test GANs
  switchTraining(metaData, true) # reenable model updating
  return metaData
end
[[GC.gc() CUDA.reclaim()] for _ in 1:2]
for lr in [1e-7]
  @show lr
  experimentMetaData = trainGANs(; opt = Flux.Optimise.Adam(lr), checkpoint = false);
  GANreport("12-25%-4-" * sciNotation(lr, 1), experimentMetaData)
end