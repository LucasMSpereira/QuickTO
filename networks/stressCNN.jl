include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")

#### Get data
if true
  @with_kw mutable struct FEAparameters
    quants::Int = 1 # number of TO problems per section
    V::Array{Real} = [0.4+rand()*0.5 for i in 1:quants] # volume fractions
    problems::Any = Array{Any}(undef, quants) # store FEA problem structs
    meshSize::Tuple{Int, Int} = (140, 50) # Size of rectangular mesh
    elementIDarray::Array{Int} = [i for i in 1:prod(meshSize)] # Vector that lists element IDs
    # matrix with element IDs in their respective position in the mesh
    elementIDmatrix::Array{Int,2} = convert.(Int, quad(meshSize...,[i for i in 1:prod(meshSize)]))
    section::Int = 1 # Number of dataset HDF5 files with "quants" samples each
  end
  FEAparams = FEAparameters()

  # @time stressCNNdata(1, 3, FEAparams)
end

### CNN

## get train data
forceData, forceMat, vm, principals = getStressCNNdata("C:/Users/LucasKaoid/Desktop/datasets/data/stressCNNdata/stressCNNdata")

## separate data for training and validation
ba = 20; sep = 0.85 # batch size and train/validation separation
# vm
vmDataTrain, vmDataValidate = splitobs((vm, forceMat); at = sep)
# principals
# prinCVdata, prinData_val = splitobs((principals, forceMat); at = sep)
# prinLoader = DataLoader(prinCVdata, batchsize = ba, shuffle = true)

## Train
# DataLoader serves to iterate in mini-batches
# of the training data of the current fold
vmTrainLoader = DataLoader((data = vmDataTrain[1], label = vmDataTrain[2]); batchsize = ba, shuffle = true, parallel = true)
vmValidateLoader = DataLoader((data = vmDataValidate[1], label = vmDataValidate[2]); batchsize = ba, shuffle = true, parallel = true)

@load "./networks/models/adam1e-6.bson" cpu_model
ssample = rand(1:size(forceData, 3))
plotVMtest(FEAparams, vm[:, :, 1, ssample], forceData[:, :, ssample], cpu_model, "modelName"; folder = "")
stressCNNtestPlots(
  10, "./networks/trainingPlots/adam1e-6 tests", vm, forceData, "adam1e-6 tests", FEAparams, cpu_model
)