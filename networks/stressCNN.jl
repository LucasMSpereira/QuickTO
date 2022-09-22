include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
### get data
forceData, forceMat, disp, forceMulti = loadDispData("C:/Users/LucasKaoid/Desktop/datasets/data/stressCNNdata/dispData"; multiOut = true)
### separate data for training, validation and test
dispTrainLoader, dispValidateLoader, dispTestLoader = getDisploaders(disp, forceMulti, 0, (0.7, 0.15), 1200; multiOutputs = true)
### Custom loss
rmseLoss(x, y) = sqrt(mean((x .- y) .^ 2)); myMultiLoss(x, y) = multiLoss(x, y; lossFun = rmseLoss)
function customLoss(x, y)
  # x: model output from one batch, y: batch of labels
  batchLoss = []
  for sampleInBatch in size(y[1], 2) # iterate inside batch
    batchLoss = vcat(batchLoss, [(x[i][:, sampleInBatch] .- y[i][:, sampleInBatch]) .^ 2 for i in axes(x)[1]] .|> mean |> mean)
  end
  return convert(Float32, mean(batchLoss)) |> gpu
end
## Grid search for architecture hyperparameters
@time hist = hyperGrid(
  multiOutputs, [3], [celu], [5], (dispTrainLoader, dispValidateLoader, dispTestLoader),
  FEAparams, customLoss, Flux.Optimise.RMSProp(1e-4); multiLossArch = true, earlyStop = 0.5)

# @load "./networks/models/3-celu-5-2.5854946E1/3-celu-5-2.5854946E1.bson" cpu_model
# gpu_model = gpu(cpu_model)
# trainmode!(gpu_model, false)
# dispCNNtestPlots(2, "./networks/models/checkpoints", dispTestLoader, "asdf", FEAparams, gpu_model, customLoss)