include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
const batchSize = 24
const nSamples = 200
# binaries for logit binary cross-entropy
const discBinaryReal = ones(Float32, batchSize)
const discBinaryFake = zeros(Float32, batchSize)

function trainGANs(; opt = Flux.Optimise.Adam())
  # object with metadata. includes instantiation of NNs,
  # optimiser, dataloaders, training configurations,
  # validation histories, and test losses
  metaData = GANmetaData(
    U_SE_ResNetGenerator(), topologyGANdisc(),
    opt, earlyStopTrainConfig(10)
  )
  @suppress_err earlyStopGANs(metaData) # train with early-stopping
  # test GANs
  metaData(GANepoch!(metaData, :test); context = :test)
  return metaData
end
[[GC.gc(true) CUDA.reclaim()] for _ in 1:2]
@time experimentMetaData = trainGANs();