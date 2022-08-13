include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")

## get train data
forceData, forceMat, vm, principals = getStressCNNdata("C:/Users/LucasKaoid/Desktop/datasets/data/stressCNNdata/stressCNNdata")
## separate data for training and validation
ba = 20 # batch size
vmTrain, vmValidate, vmTest = splitobs((vm[:, :, :, 1:1000], forceMat[:, 1:1000]); at = (0.7, 0.15))
# vmTrain, vmValidate, vmTest = splitobs((vm, forceMat); at = (0.7, 0.15))
## Train
# DataLoader serves to iterate in mini-batches
# of the training data of the current fold
vmTrainLoader = DataLoader((data = vmTrain[1], label = vmTrain[2]); batchsize = ba, shuffle = true, parallel = true);
vmValidateLoader = DataLoader((data = vmValidate[1], label = vmValidate[2]); batchsize = ba, shuffle = true, parallel = true);
vmTestLoader = DataLoader((data = vmTest[1], label = vmTest[2]); batchsize = ba, shuffle = true, parallel = true);
# Grid search for architecture hyperparameters
hist = hyperGrid(
  convMaxPoolBN, [3, 5], [relu, swish], [5, 10],
  (vmTrainLoader, vmValidateLoader, vmTestLoader), FEAparams
)