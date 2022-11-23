# Functions that organize data in loaders. These enable the iteration
# in batches of data.

# prepare data loaders for loadCNN training/validation/test using displacements
function getDisploaders(disp, forces, numPoints, separation, batch; multiOutputs = false)
  if numPoints != 0 # use all data or certain number of points
    if !multiOutputs
      dispTrain, dispValidate, dispTest = splitobs((disp[:, :, :, 1:numPoints], forces[:, 1:numPoints]); at = separation, shuffle = true)
    else
      dispTrain, dispValidate, dispTest = splitobs((disp[:, :, :, 1:numPoints],
          (forces[1][:, 1:numPoints], forces[2][:, 1:numPoints],
          forces[3][:, 1:numPoints], forces[4][:, 1:numPoints]));
          at = separation, shuffle = true)
    end
  else
    dispTrain, dispValidate, dispTest = splitobs((disp, forces); at = separation, shuffle = true)
  end
  return ( # DataLoader serves to iterate in mini-batches of the training data
    DataLoader((data = dispTrain[1], label = dispTrain[2]); batchsize = batch, parallel = true),
    DataLoader((data = dispValidate[1], label = dispValidate[2]); batchsize = batch, parallel = true),
    DataLoader((data = dispTest[1], label = dispTest[2]); batchsize = batch, parallel = true))
end

# prepare data loaders for loadCNN training/validation/test in FEAloss pipeline
function getFEAlossLoaders(
  disp::Array{Float32, 4}, sup::Array{Float32, 3}, vf::Array{Float32, 1},
  force, numPoints::Int64, separation, batch::Int64
)
  # split dataset into training, validation, and testing
  numPoints == 0 && (numPoints = length(vf))
  numPoints < 0 && (error("Negative number of samples in getFEAlossLoaders."))
  FEAlossTrain, FEAlossValidate, FEAlossTest = splitobs(
    (
      disp[:, :, :, 1:numPoints],
      sup[:, :, 1:numPoints], vf[1:numPoints],
      (force[1][:, 1:numPoints], force[2][:, 1:numPoints],
      force[3][:, 1:numPoints], force[4][:, 1:numPoints])
    ); at = separation, shuffle = true)
  # return loaders for each stage
  return (DataLoader(FEAlossTrain; batchsize = batch, parallel = true),
    DataLoader(FEAlossValidate; batchsize = batch, parallel = true),
    DataLoader(FEAlossTest; batchsize = batch, parallel = true))
end

# prepare data loaders for loadCNN training/validation/test
function getVMloaders(vm, forces, numPoints, separation, batch; multiOutputs = false)
  # use all data or certain number of points
  if numPoints != 0
    if !multiOutputs
      vmTrain, vmValidate, vmTest = splitobs((vm[:, :, :, 1:numPoints], forces[:, 1:numPoints]); at = separation, shuffle = true)
    else
      vmTrain, vmValidate, vmTest = splitobs(
        (vm[:, :, :, 1:numPoints], (forces[1][:, 1:numPoints], forces[2][:, 1:numPoints], forces[3][:, 1:numPoints], forces[4][:, 1:numPoints]));
        at = separation, shuffle = true)
    end
  else
    vmTrain, vmValidate, vmTest = splitobs((vm, forces); at = separation, shuffle = true)
  end
  # DataLoader serves to iterate in mini-batches of the training data
  vmTrainLoader = DataLoader((data = vmTrain[1], label = vmTrain[2]); batchsize = batch, parallel = true);
  vmValidateLoader = DataLoader((data = vmValidate[1], label = vmValidate[2]); batchsize = batch, parallel = true);
  vmTestLoader = DataLoader((data = vmTest[1], label = vmTest[2]); batchsize = batch, parallel = true);
  return vmTrainLoader, vmValidateLoader, vmTestLoader
end