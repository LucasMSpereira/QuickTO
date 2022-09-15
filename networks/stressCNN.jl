include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
Random.seed!(3111);
### get data
# forceData, forceMat, vm, principals, forceMulti = getStressCNNdata("C:/Users/LucasKaoid/Desktop/datasets/data/stressCNNdata/stressCNNdata"; multiOut = true)
@time forceData, forceMat, disp, forceMulti = loadDispData("C:/Users/LucasKaoid/Desktop/datasets/data/stressCNNdata/dispData"; multiOut = true)
### separate data for training, validation and test
# vmTrainLoader, vmValidateLoader, vmTestLoader = getVMloaders(vm, forceMulti, 0, (0.7, 0.15), 60; multiOutputs = true)
dispTrainLoader, dispValidateLoader, dispTestLoader = getDisploaders(disp, forceMulti, 1000, (0.7, 0.15), 60; multiOutputs = true)
# Grid search for architecture hyperparameters
rmseLoss(x, y) = sqrt(mean((x .- y) .^ 2))
myMultiLoss(x, y) = multiLoss(x, y; lossFun = rmseLoss)
@time hist = hyperGrid(
multiOutputs, [13], [celu], [5], (dispTrainLoader, dispValidateLoader, dispTestLoader),
FEAparams, myMultiLoss, Flux.Optimise.RMSProp(2.5e-6); multiLossArch = true, earlyStop = 10.0)