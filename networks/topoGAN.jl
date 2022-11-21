include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
const batchSize = 64
const nSamples = 12_000
# binaries for logit binary cross-entropy
const discBinaryReal = ones(Float32, batchSize)
const discBinaryFake = zeros(Float32, batchSize)
function earlyStopGANs(metaData)
  epoch = 0
  while true # loop in epochs
    epoch += 1 # count training epochs
    epoch == 1 && println("Epoch       Generator loss    Discriminator loss")
    # trains for one epoch
    GANepoch!(metaData, :train)
    # occasionally run validation epoch and check for early-stopping
    if epoch % metaData.trainConfig.validFreq == 0
      switchTraining(metaData, false) # disable model updating during validation
      # validation epoch returning avg losses for both NNs
      GANepoch!(metaData, :validate) |> metaData
      # print information about validation
      switchTraining(metaData, true) # reenable model updating after validation
      # after enough validations, start performing early-stop check
      if length(metaData.lossesVals[:genValHistory]) > metaData.trainConfig.earlyStopQuant
        # percentage drops in losses and boolean indicating to stop training
        genValLossPercentDrop, discValLossPercentDrop, earlyStopping = earlyStopCheck(metaData)
        # print validation and early-stop information
        GANprints(epoch, metaData; earlyStopVals = (genValLossPercentDrop, discValLossPercentDrop))
        if earlyStopping
          println("EARLY-STOPPING")
          saveGANs(metaData; finalSave = true)
          break
        end
      else # not checking for early-stop yet
        GANprints(epoch, metaData) # print just validation information
      end
    end
    # save occasional checkpoints of the model
    epoch % metaData.trainConfig.checkPointFreq == 0 && saveGANs(metaData)
  end
end
function trainGANs(; opt = Flux.Optimise.Adam())
  # object with metadata. includes instantiation of NNs,
  # optimiser, dataloaders, training configurations,
  # validation histories, and test losses
  metaData = GANmetaData(
    U_SE_ResNetGenerator(), topologyGANdisc(), opt,
    createGANloaders(GANdata()..., batchSize; separation = (0.7, 0.2)),
    earlyStopTrainConfig(2)
  )
  @suppress_err earlyStopGANs(metaData) # train with early-stopping
  # test GANs
  GANepoch!(metaData, :test) |> x -> metaData(x; context = :test)
  return metaData
end
[[GC.gc(true) CUDA.reclaim()] for _ in 1:2]
@time experimentMetaData = trainGANs();