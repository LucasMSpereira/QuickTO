include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
Random.seed!(3111);
### get data
# forceData, forceMat, vm, principals = getStressCNNdata("C:/Users/LucasKaoid/Desktop/datasets/data/stressCNNdata/stressCNNdata")
forceData, forceMat, vm, principals, forceMulti = getStressCNNdata("C:/Users/LucasKaoid/Desktop/datasets/data/stressCNNdata/stressCNNdata"; multiOut = true)
### separate data for training, validation and test
# vmTrainLoader, vmValidateLoader, vmTestLoader = getLoaders(vm/maximum(vm), forceMat/maxForceMat, 0, (0.7, 0.15), 60)
# vmTrainLoader, vmValidateLoader, vmTestLoader = getLoaders(vm, forceMat, 0, (0.7, 0.15), 60)
vmTrainLoader, vmValidateLoader, vmTestLoader = getLoaders(vm, forceMulti, 0, (0.7, 0.15), 60; multiOutputs = true)
# Grid search for architecture hyperparameters
rmseLoss(x, y) = sqrt(mean((x .- y) .^ 2))
myMultiLoss(x, y) = multiLoss(x, y; lossFun = rmseLoss)
# @time hist = hyperGrid(
# multiOutputs, [13], [celu], [5, 20, 40], (vmTrainLoader, vmValidateLoader, vmTestLoader),
# FEAparams, myMultiLoss, Flux.Optimise.RMSProp(2.5e-6); multiLossArch = true)
@load "./networks/models/13-celu-5-3.3333335E1/13-celu-5-3.3333335E1.bson" cpu_model
gpu_model = gpu(cpu_model)
# PDF with list of hyperparameters
parameterList(gpu_model, Flux.Optimise.RMSProp(2.5e-6), myMultiLoss, "./networks/models/13-celu-5-3.3333335E1"; multiLossArch = true)
# Create test plots and unite all PDF files in folder
testTargets = shapeTargetStressCNN(true, vmTestLoader)
stressCNNtestPlots(
  20, "./networks/models/13-celu-5-3.3333335E1",
  vmTestLoader.data.data.parent, testTargets, "13-celu-5-3.3333335E1", FEAparams, gpu_model, myMultiLoss
)