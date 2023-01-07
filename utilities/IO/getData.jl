# functions that read data from hdf5 files, usually to be
# used for ML pipelines

# prepare data for GAN training. Receives list of HDF5 files.
# Returns data loader to iterate in batches of samples
function GANdata(metaData::GANmetaData, group, goal::Symbol, _lastFileBatch::Bool)
  genInput = zeros(Float32, FEAparams.meshMatrixSize..., 3, 1); FEAinfo = similar(genInput)
  topology = zeros(Float32, FEAparams.meshMatrixSize..., 1, 1)
  if goal != :test
    if _lastFileBatch
      if length(metaData.files[goal][group]) > 1
        for file in Iterators.take(metaData.files[goal][group], length(metaData.files[goal][group]) - 1)
          dataDict = readTopologyGANdataset(file) # read data as dictionary
          genInput, FEAinfo, topology = groupGANdata!(
            genInput, FEAinfo, topology, dataDict
          )
        end
      end
      # number of samples to read from last file
      sampleAmount = contextQuantity(goal, metaData.files[goal], metaData.datasetUsed)
      dataDict = readTopologyGANdataset(metaData.files[goal][group] |> last)
      genInput, FEAinfo, topology = groupGANdata!(
        genInput, FEAinfo, topology, dataDict;
        sampleAmount = sampleAmount
      )
    else
      for file in metaData.files[goal][group]
        dataDict = readTopologyGANdataset(file) # read data as dictionary
        genInput, FEAinfo, topology = groupGANdata!(
          genInput, FEAinfo, topology, dataDict
        )
      end
    end
  else
    # number of samples to read from last file
    sampleAmount = contextQuantity(goal, group, metaData.datasetUsed)
    dataDict = readTopologyGANdataset(group[1]) # read data as dictionary
    genInput, FEAinfo, topology = groupGANdata!(
      genInput, FEAinfo, topology, dataDict;
      sampleAmount = sampleAmount
    )
  end
  # discard first position of arrays (initialization)
  genInput, FEAinfo, topology = remFirstSample.((genInput, FEAinfo, topology))
  if normalizeDataset # optional data normalization in [-1; 1]
    genInput = mapslices(normalizeVals, genInput; dims = [1 2])
    FEAinfo = mapslices(normalizeVals, FEAinfo; dims = [1 2])
    topology = mapslices(normalizeVals, topology; dims = [1 2])
  end
  return DataLoader( # return data loader to iterate in batches
    (
      genInput, # generator input
      FEAinfo, # FEA data
      topology # true topologies
    ); batchsize = batchSize, parallel = true
  )
end

# Get data from dataset file
function getDataFSVDT(file)
  id = h5open(file, "r")
  top = read(id["topologies"])
  forces = read(id["inputs"]["forces"])
  supps = read(id["inputs"]["dispBoundConds"])
  vf = read(id["inputs"]["VF"])
  disp = read(id["disp"])
  close(id)
  return forces, supps, vf, disp, top
end

# Create displacements dataset by gathering that info in new file
function getDispData(path)
  files = glob("*", path)
  count = 0 # global sample counter
  # initialize "global" variables
  globalF = zeros(2, 4, 1); globalDisp = zeros(51, 141, 1)
  for file in keys(files) # loop in files
    id = h5open(files[file], "r") # open current file
    force = read(id["forces"]); disp = read(id["disp"]) # read data
    close(id) # close current file
    # concatenate file data with "global" arrays
    globalF = cat(globalF, force; dims = 3)
    globalDisp = cat(globalDisp, disp; dims = 3)
    count += size(force, 3) # update global counter
    @show count
  end
  # discard null initial data
  globalF = globalF[:, :, 2:end]; globalDisp = globalDisp[:, :, 2:end]
  # create new file to store everything
  new = h5open(datasetPath*"data/stressCNNdata/dispData", "w")
  # initialize fields in new file
  create_dataset(new, "forces", zeros(2, 4, count))
  create_dataset(new, "disp", zeros(51, 141, 2*count))
  # fill new file with data
  for sample in 1:count
    new["forces"][:, :, sample] = globalF[:, :, sample]
    new["disp"][:, :, 2*sample-1 : 2*sample] = globalDisp[:, :, 2*sample-1 : 2*sample]
  end
  close(new) # close and save new file
end

#= create dataset to be used in new pipeline of load prediction through
displacements. Dataset contains displacements, VF ,support, and load
of samples. Initially restricted to samples with left side clamped. =#
function getFEAlossData(id)
  # get list of file names in folder "id"
  fileList = readdir(datasetPath*"data/$id"; join = true, sort = false)
  count = 0 # global sample counter
  # initialize "global" variables
  globalVF = zeros(1); globalDisp = zeros(51, 141, 1)
  globalSup = zeros(3, 3, 1); globalForce = zeros(2, 4, 1)
  for file in fileList # loop in files
    force, sup, vf, disp, _ = getDataFSVDT(file) # read data
    samples = findall(x -> x == 4, sup[1, 3, :]) # find samples with desired support type
    # concatenate subset of file data with "global" arrays
    globalVF = vcat(globalVF, vf[samples])
    globalForce = cat(globalForce, force[:, :, samples]; dims = 3)
    globalSup = cat(globalSup, sup[:, :, samples]; dims = 3)
    [globalDisp = cat(globalDisp, disp[:, :, 2*s-1 : 2*s]; dims = 3) for s in samples]
    count += length(samples) # update global counter
    println("count = $count - ", findfirst(x -> x == file, fileList), "/", length(fileList))
  end
  # discard null initial data
  globalVF = globalVF[2:end]; globalDisp = globalDisp[:, :, 2:end]
  globalSup = globalSup[:, :, 2:end]; globalForce = globalForce[:, :, 2:end]
  # create new file to store everything
  new = h5open(datasetPath*"data/stressCNNdata/fea loss data/FEAlossData$id", "w")
  # initialize fields in new file
  create_dataset(new, "vf", zeros(count))
  create_dataset(new, "disp", zeros(51, 141, 2*count))
  create_dataset(new, "sup", zeros(3, 3, count))
  create_dataset(new, "force", zeros(2, 4, count))
  # fill new file with data
  for i in 1:count
    new["vf"][i] = globalVF[i]
    new["sup"][:, :, i] = globalSup[:, :, i]
    new["disp"][:, :, 2*i-1 : 2*i] = globalDisp[:, :, 2*i-1 : 2*i]
    new["force"][:, :, i] = globalForce[:, :, i]
  end
  close(new) # close and save new file
end

# load vm and force dataset from file
function getStressCNNdata(path; multiOut = false)
  h5file = h5open(path, "r") # open hdf5 file
  datasets = HDF5.get_datasets(h5file) # get references to datasets
  # read force data (2x4 Float matrix per sample)
  forceData = convert.(Float32, HDF5.read(datasets[1])) # 2 x 4 x nSamples of Float32
  # reshape forces to 8 x nSamples float matrix. each col refers to a sample
  forceMat = hcat([vec(reshape(forceData[:, :, i], (1, :))) for i in axes(forceData)[3]]...)
  # Get VM data, reshape to 50 x 140 x 1 x nSamples and convert to Float32
  vm = convert.(Float32, reshape(HDF5.read(datasets[3]), 50, 140, 1, :))
  prin = HDF5.read(datasets[2])
  principals = Array{Any}(undef, size(vm, 3))
  [principals[c] = prin[:, :, 2*c-1 : 2*c] for c in 1:size(vm, 3)]
  close(h5file)
  # reshape force data for models with multiple outputs
  if multiOut
    xPositions = zeros(Float32, (2, size(forceData, 3)))
    yPositions = similar(xPositions)
    firstComponents = similar(xPositions)
    secondComponents = similar(xPositions)
    for sample in axes(forceData)[3]
      xPositions[:, sample] .= forceData[:, 1, sample]
      yPositions[:, sample] .= forceData[:, 2, sample]
      firstComponents[:, sample] .= forceData[:, 3, sample]
      secondComponents[:, sample] .= forceData[:, 4, sample]
    end
    return forceData, forceMat, vm, principals, (xPositions, yPositions, firstComponents, secondComponents)
  end
  return forceData, forceMat, vm, principals
end

# group certain data used to train GANs
function groupGANdata!(genInput, FEAinfo, topology, dataDict; sampleAmount = 0)
  # amount of samples according to percentage of dataset being used
  if sampleAmount == 0
    nSamples = length(dataDict[:compliance])
  else
    nSamples = sampleAmount
  end
  standardSize = (FEAparams.meshMatrixSize..., 1, nSamples)
  genInput = cat(genInput, # generator input
      solidify(
        reshape(dataDict[:vf][:, :, 1, 1:nSamples], standardSize),
        reshape(dataDict[:vm][:, :, 1, 1:nSamples], standardSize),
        reshape(dataDict[:energy][:, :, 1, 1:nSamples], standardSize)
      );
      dims = 4
  )
  FEAinfo = cat(FEAinfo, # FEA conditioning info
    solidify(
      reshape(dataDict[:binarySupp][:, :, 1, 1:nSamples], standardSize),
      reshape(dataDict[:Fx][:, :, 1, 1:nSamples], standardSize),
      reshape(dataDict[:Fy][:, :, 1, 1:nSamples], standardSize)
    );
    dims = 4
  )
  topology = cat(topology, # REAL topologies
    reshape(dataDict[:topologies][:, :, 1, 1:nSamples], standardSize); dims = 4
  )
  return genInput, FEAinfo, topology
end

# load displacement dataset from file
function loadDispData(multiOut)
  forceData = []; forceMat = []; disp = []
  h5open(datasetPath*"data/stressCNNdata/dispData", "r") do h5file # open hdf5 file
    datasets = HDF5.get_datasets(h5file) # get references to datasets
    # read force data (2x4 Float matrix per sample)
    forceData = convert.(Float32, HDF5.read(datasets[2])) # 2 x 4 x nSamples of Float32
    # reshape forces to 8 x nSamples float matrix. each col refers to a sample
    forceMat = hcat([vec(reshape(forceData[:, :, i], (1, :))) for i in axes(forceData)[3]]...)
    # Get displacement data
    dispRead = HDF5.read(datasets[1]) # 51 x 141 x 2*nSamples of Float32
    disp = Array{Any}(undef, (51, 141, 2, size(forceData, 3)))
    [disp[:, :, :, sample] .= dispRead[:, :, 2*sample-1 : 2*sample] for sample in axes(forceData)[3]]
  end
  # reshape force data for models with multiple outputs
  if multiOut
    xPositions = zeros(Float32, (2, size(forceData, 3)))
    yPositions = similar(xPositions)
    firstComponents = similar(xPositions)
    secondComponents = similar(xPositions)
    for sample in axes(forceData)[3]
      xPositions[:, sample] .= forceData[:, 1, sample]
      yPositions[:, sample] .= forceData[:, 2, sample]
      firstComponents[:, sample] .= forceData[:, 3, sample]
      secondComponents[:, sample] .= forceData[:, 4, sample]
    end
    return forceData, forceMat, convert.(Float32, disp), (xPositions, yPositions, firstComponents, secondComponents)
  end
  return forceData, forceMat, convert.(Float32, disp)
end

# read data to be used in FEAloss ML pipeline
function loadFEAlossData()
  disp, sup, vf, force, xPositions, yPositions, firstComponents, secondComponents = [], [], [], [], [], [], [], []
  h5open(datasetPath*"data/stressCNNdata/FEAlossData", "r") do h5file # open hdf5 file
    datasets = HDF5.get_datasets(h5file) # get references to datasets
    # read displacement data (51 x 141 x nSamples)
    dispRead = HDF5.read(datasets[1])
    # reshape displacements to match Flux.jl's API
    disp = Array{Float32}(undef, (51, 141, 2, size(dispRead, 3) ÷ 2))
    [disp[:, :, :, sample] .= dispRead[:, :, 2*sample-1 : 2*sample] for sample in 1 : size(dispRead, 3)÷2]
    force = convert.(Float32, HDF5.read(datasets[2])) # read force data (2 x 4 x nSamples)
    # reshape force data
    xPositions = zeros(Float32, (2, size(force, 3)))
    yPositions = similar(xPositions)
    firstComponents = similar(xPositions)
    secondComponents = similar(xPositions)
    for sample in axes(force)[3] # loop in samples
      xPositions[:, sample] .= force[:, 1, sample]
      yPositions[:, sample] .= force[:, 2, sample]
      firstComponents[:, sample] .= force[:, 3, sample]
      secondComponents[:, sample] .= force[:, 4, sample]
    end
    # read supports data (3 x 3 x nSamples)
    sup = convert.(Float32, HDF5.read(datasets[3]))
    # read vf data (nSamples vector)
    vf = convert.(Float32, HDF5.read(datasets[4]))
  end
  return disp, sup, vf, (xPositions, yPositions, firstComponents, secondComponents)
end