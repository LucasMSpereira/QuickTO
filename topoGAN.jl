import Pkg
Pkg.activate(".")
Pkg.instantiate()
# setup base path for dataset (ends with "datasets/")
# datasetPath = ""
# setup base path for project (ends with "QuickTO/")
# projPath = ""
include(projPath * "QTOutils.jl")

const desktop = true
const batchSize = 64
const normalizeDataset = false # choose to normalize data in [-1; 1]
const startTime = timeNow()
const percentageDataset = 0.6 # fraction of dataset to be used
const wasserstein = true

# Start straining with specific configuration. After a certain number of training
# epochs, a validation epoch is done. After all training and validation, a test
# epoch is performed. Model checkpoints are occasionally saved. The final models
# are saved as well.
@time expMetaData = trainGANs(;
  ## Choose optimizers for generator and discriminator.
  genOpt_ = Flux.Optimise.AdamW(1e-4),
  discOpt_ = Flux.Optimiser(Flux.ClipNorm(1.0), Flux.Optimise.AdamW(1e-4)),
  ## (optional) If not training from scratch, uncomment lines of next 4
  ## keyword arguments and provide names of pre-trained models, alongside
  ## the paths to the metadata file and the original folder of the model.
  # genName_ = "01-30T13-49-30-3gen.bson",
  # discName_ = "01-30T13-50-04-3disc.bson",
  # metaDataName = projPath * "networks/GANplots/01-29T09-45-03-Bvp4/01-29T20-07-39metaData.txt",
  # originalFolder = projPath * "networks/GANplots/01-29T09-45-03-Bvp4/",
  ## Determine architectures to be used for each network. Their definitions
  ## are in "./QuickTO/utilities/ML utils/architectures.jl".
  architectures = (
    # convNextModel(96, [3, 3, 9, 3], 0.5),
    # convNextModel(128, [3, 3, 27, 3], 0.5),
    convNextModel(192, [3, 3, 27, 3], 0.5),
    # U_SE_ResNetGenerator(),
    # patchGANdisc()
    topologyGANdisc(; drop = 0.3)
  ),
  ## Define training configurations. Only total number of epochs,
  ## and validation interval are required. Definition is
  ## in ./QuickTO/utilities/typeDefinitions.jl
  trainConfig = epochTrainConfig(3, 3)
  # trainConfig = earlyStopTrainConfig(
  #   1; earlyStopQuant = 2, earlyStopPercent = 5
  # )
)
saveGANs(expMetaData, 0; finalSave = true) # save final models
switchTraining(expMetaData, false)
GANreport(expMetaData) # create report