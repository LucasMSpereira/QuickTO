include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")

#### Get data
if false
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

  @time stressCNNdata(1, 3, FEAparams)
end

### CNN

## get train data
h5file = h5open("C:/Users/LucasKaoid/Desktop/datasets/data/stressCNNdata/stressCNNdata", "r") # open hdf5 file
datasets = HDF5.get_datasets(h5file) # get datasets references
forceData = HDF5.read(datasets[1]) # read force data (2x4 Float matrix per sample)
# reshape to 8xN float matrix. each col refers to a sample
forceMat = convert.(Float32, hcat([vec(reshape(forceData[:, :, i], (1, :))) for i in 1:size(forceData, 3)]...))
# forceMat = standardize(ZScoreTransform, forceMat; dims = 2)
vm = convert.(Float32, reshape(HDF5.read(datasets[3]), 50, 140, 1, :))
# for line in 1:size(vm, 1)
#   for col in 1:size(vm, 2)
#     vm[line, col, 1, :] .= standardize(ZScoreTransform, vm[line, col, 1, :]; dims = 1)
#   end
# end
# prin = HDF5.read(datasets[2])
# principals = Array{Any}(undef, size(vm, 3))
# [principals[c] = prin[:, :, 2*c-1 : 2*c] for c in 1:size(vm, 3)]
close(h5file)
## separate data for training and validation
ba = 10; sep = 0.85 # batch size and train/validation separation
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
# Training parameters

rates = [1e-4, 1e-6, 1e-8]
params = [epochTrainConfig(; epochs = 30, schedule = 0, decay = 0, evalFreq = 2, evaluations = []) for i in 1:length(rates)]
testRates!(rates, params)
plotLearnTries(params, rates; drawLegend = true)
mean(params[3].evaluations)