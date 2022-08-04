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
forceMat = standardize(ZScoreTransform, forceMat; dims = 2)
vm = convert.(Float32, reshape(HDF5.read(datasets[3]), 50, 140, 1, :))
for line in 1:size(vm, 1)
  for col in 1:size(vm, 2)
    vm[line, col, 1, :] .= standardize(ZScoreTransform, vm[line, col, 1, :]; dims = 1)
  end
end
prin = HDF5.read(datasets[2])
principals = Array{Any}(undef, size(vm, 3))
[principals[c] = prin[:, :, 2*c-1 : 2*c] for c in 1:size(vm, 3)]
close(h5file)
## separate data for training and validation
ba = 60; sep = 0.85 # batch size and train/validation separation
# vm
vmCVdata, vmData_val = splitobs((vm, forceMat); at = sep)
# principals
# prinCVdata, prinData_val = splitobs((principals, forceMat); at = sep)
# prinLoader = DataLoader(prinCVdata, batchsize = ba, shuffle = true)

if false
  function f(k, channel, activ)
    global dd += 1
    model = Chain(
      Conv((k, k), 1 => channel, activ)
    )
    obj = mean(model(mat))
    println("$dd $channel $activ $k $obj")
    return obj
  end

  mat = rand(50, 50, 1, 1)
  begin
    dd = 0
    HOHB = @hyperopt for i = 10,
        # List of possibilities of parameters
        sampler = Hyperband(R=50, η=3, inner = BOHB()),
        channel = [1, 2],
        activ = [relu, selu, swish],
        k = [2, 3]
        
      ob = vec([f(k, channel, activ)])

    end
  end
end

## Train

#= k-fold cross-validation: number of folds in which training/validation
dataset will be split. In each of the k iterations, (k-1) folds will be used for
training, and the remaining fold will be used for validation =#
  stressCNN, modParams = buildModel(relu) # build model
  
  @showprogress "Epoch" for epoch in 1:80
    # DataLoader serves to iterate in mini-batches
    # of the training data of the current fold
    vmTrainLoader = DataLoader((data = fold[1][1], label = fold[1][2]); batchsize = ba, parallel = true)
    # For each training mini-batch, calculate gradient and update parameters
    for (x, y) in vmTrainLoader
      grads = Flux.gradient(modParams) do
        Flux.mae(stressCNN(gpu(x)), gpu(y))
      end
      Flux.Optimise.update!(Adam(), modParams, grads)
    end
  end
  cpu_model = cpu(stressCNN) # copy model of current fold to cpu
  # Mean of prediction loss in validation set of current fold
  meanEval = mean(Flux.mae(cpu_model(fold[2][1]), fold[2][2]))
  println("Fold validation: $(meanEval)")
  # Save model including mean evaluation loss in the BSON file's name
  # BSON.@save "model-$(replace(string(round( meanEval; digits = 6 )), "." => "-")).bson" cpu_model
  # BSON.@save "model-$(replace(string(ceil(now(), Dates.Second)), ":"=>"-")).bson" cpu_model
end