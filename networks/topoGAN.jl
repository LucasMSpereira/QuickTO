include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
const batchSize = 64
# binaries for logit binary cross-entropy
const discBinaryReal = ones(Float32, batchSize)
const discBinaryFake = zeros(Float32, batchSize)
percentageDataset::Float64 = 0.29

function trainGANs(; opt = Flux.Optimise.Adam())
  # object with metadata. includes instantiation of NNs,
  # optimiser, dataloaders, training configurations,
  # validation histories, and test losses
  metaData = GANmetaData(
    # U_SE_ResNetGenerator(), topologyGANdisc(),
    loadGANs()...,
    opt, epochTrainConfig(5, 5)
  )
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
[[GC.gc(true) CUDA.reclaim()] for _ in 1:2]
# @time experimentMetaData = trainGANs();
GANreport("teste", experimentMetaData)
# begin
  #   m = GANmetaData(
  #     Chain(Conv((3, 3), 1 => 1)), Chain(Conv((3, 3), 1 => 1)),
  #     Flux.Optimise.Adam(), epochTrainConfig(67, 5)
  #   )
  #   [m(losses[l]) for l in axes(losses, 1)]
  #   m((randBetween(100, 1000), randBetween(0, 5)); context = :test)
  #   GANreport("modelName", m)
  #   writeLosses(m)
# end

