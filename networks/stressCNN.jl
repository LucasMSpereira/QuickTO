include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
### get data
forceData, forceMat, disp, forceMulti = loadDispData("C:/Users/LucasKaoid/Desktop/datasets/data/stressCNNdata/dispData"; multiOut = true)
disp ./= maximum(abs, disp) # normalize in [-1.0; 1.0]
forceMulti[3] .+= 90; forceMulti[4] .+= 90 # shift force components from [-90; 90] to [0; 180]
### separate data for training, validation and test
dispTrainLoader, dispValidateLoader, dispTestLoader = getDisploaders(disp, forceMulti, 8000, (0.7, 0.15), 1200; multiOutputs = true)
### Custom loss
function customLoss(x, y)
  # x: model output from one batch, y: batch of labels
  batchLoss = []
  for sampleInBatch in 1:size(y[1], 2) # iterate inside batch
    batchLoss = vcat(batchLoss, [(x[i][:, sampleInBatch] - y[i][:, sampleInBatch]) .^ 2 for i in axes(x)[1]] .|> mean |> mean)
  end
  return convert(Float32, mean(batchLoss)) |> gpu
end
## Grid search for architecture hyperparameters
@time hyperGrid(multiOutputs, [5], [celu], [5], (dispTrainLoader, dispValidateLoader, dispTestLoader),
  FEAparams, customLoss, Flux.Optimise.NAdam(1e-4); multiLossArch = true, earlyStop = 1)

#
