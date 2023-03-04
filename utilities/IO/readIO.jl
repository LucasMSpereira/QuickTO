# functions focused on only reading data from files

# read hdf5 file from GAN dataset and represent
# it in "dense" form. Many utility functions
# are based on these "dense" forms
function denseInfoFromGANdataset(path::String, sampleAmount::Int)::Tuple{Dict, Dict{Symbol, Array{Float32}}}
  # keys: compliance, vf, vm, energy, binarySupp, Fx, Fy, topologies
  dataDict = readTopologyGANdataset(path) # data from file
  compliance = dataDict[:compliance]::Vector{Float32}
  # choose random 'numSamples' samples in file
  sampleIDs = shuffle(1:length(dataDict[:compliance]))[1:sampleAmount]
  # transform data of desired samples to dense format
  vf = (dataDict[:vf][1, 1, :, sampleIDs] |> vec)::Vector{Float32}
  vm = Array{Matrix{Float32}}(undef, length(sampleIDs))
  energy = Array{Matrix{Float32}}(undef, length(sampleIDs))
  for (index, sample) in enumerate(sampleIDs)
    vm[index] = linear_interpolation((nodeY, nodeX), dataDict[:vm][:, :, 1, sample])(centroidY, centroidX)
    energy[index] = linear_interpolation((nodeY, nodeX), dataDict[:energy][:, :, 1, sample])(centroidY, centroidX)
  end
  denseSupport = binaryToDenseSupport(dataDict[:binarySupp][:, :, 1, sampleIDs])
  force = forceMatrixToDense(dataDict[:Fx][:, :, 1, sampleIDs], dataDict[:Fy][:, :, 1, sampleIDs])
  smallDataDict = Dict()
  for data in keys(dataDict)
    if ndims(dataDict[data]) == 4
      push!(smallDataDict, data => dataDict[data][:, :, 1, sampleIDs])
    else
      push!(smallDataDict, data => dataDict[data][sampleIDs])
    end
  end
  return (
    Dict(
      :compliance => compliance[sampleIDs],
      :vf => vf,
      :vm => vm,
      :energy => energy,
      :denseSupport => denseSupport,
      :force => force,
      :topologies => dataDict[:topologies][1:50, 1:140, 1, sampleIDs]
    ),
    smallDataDict
  ) 
end

# get lists of hdf5 files to be used in training and validation
# using "lowest upper bound" logic across files
function getNonTestFileLists(trainValidateFolder, trainPercentage)
  filePaths = readdir(trainValidateFolder; join = true)
  trainSplit = 1; validateSplit = 1
  for i in keys(filePaths) # loop in files
    # if amount of samples is less then the desired
    # amount for training
    if numSample(
      filePaths[1 : max(1, i - 1)]
    ) <= datasetNonTestSize * 0.7 * percentageDataset
      trainSplit = i
      validateSplit = copy(trainSplit) + 1
    end
    # if training files have been determined
    if i > trainSplit
      # if amount of samples is less then the desired
      # amount for validating
      if numSample(
        filePaths[trainSplit + 1 : min(length(filePaths), validateSplit)]
      ) <= datasetNonTestSize * 0.3 * percentageDataset
        validateSplit = i
      else
        break
      end
    end
  end
  return Dict( # create lists of files to be used in each split
    :train => filePaths[1:trainSplit],
    :validate => filePaths[trainSplit + 1 : validateSplit]
  )
end

# get validation histories, test losses and
# validation frequency from txt report for plotting
function getValuesFromTxt(txtPath::String)
    content = readlines(txtPath)
    # line with heading of validation histories
    heading = findfirst(==("EPOCH   GENERATOR      DISCRIMINATOR"), content)
    # line with last validation
    finalValidation = findnext(==(""), content, heading) - 1
    genHist = zeros(Float32, finalValidation - heading)
    discHist = zeros(Float32, finalValidation - heading)
    for (index, line) in enumerate(heading + 1:finalValidation)
        s = split(content[line])
        genHist[index] = parse(Float32, s[2])
        discHist[index] = parse(Float32, s[3])
    end
    # line with header for test loss values
    testHeader = findfirst(==("********* TEST LOSSES"), content)
    if testHeader === nothing # no testing phase
        testLosses = (0.0, 0.0)
    else # get test values
        testLosses = (
        parse(Float32, split(content[testHeader + 2])[2]),
        parse(Float32, split(content[testHeader + 3])[2])
        )
    end
    s = split.(content) |> Iterators.flatten |> collect # all "words"
    valFreq = 0
    for element in 1:length(s) - 1 # get validation frequency
        s[element : element + 1] == ["VALIDATION", "FREQUENCY"] && (valFreq = parse(Int32, s[element + 3]))
    end
    return genHist, discHist, testLosses, valFreq
end

# inspect contents of HDF5 file
function HDF5inspect(HDF5path)
  h5file = h5open(HDF5path, "r") # open file
  datasets = HDF5.get_datasets(h5file) # datasets in file
  data = HDF5.read.(datasets) # get file data as vector of contents of datasets
  for ds in axes(data)[1]
    println("\nDataset $ds $(HDF5.name(datasets[ds])[2:end]) $(size(data[ds]))\n")
    statsum(data[ds])
  end
  close(h5file)
end

# load previous GAN models
function loadGANs(genName, discName)
  gen = convNextModel(192, [3, 3, 27, 3], 0.5)
  disc = topologyGANdisc(; drop = 0.3)
  BSON.@load datasetPath * "data/checkpoints/" * genName cpuGenerator
  BSON.@load datasetPath * "data/checkpoints/" * discName cpuDiscriminator
  return (
    Flux.loadmodel!(gen, cpuGenerator),
    Flux.loadmodel!(disc, cpuDiscriminator),
  )
end

# load saved generator
function loadGenerator(genName)
  BSON.@load datasetPath * "data/checkpoints/" * genName cpuGenerator
  return cpuGenerator |> gpu
end

# read data from global analysis file generated by combineFiles()
function readAnalysis(pathRef)
    file = h5open(datasetPath*"analyses/"*pathRef, "r")
    ds = read(file["dataset"])
    res = read(file["result"])
    sID = read(file["sampleID"])
    sec = read(file["section"])
    close(file)
    return ds, sec, sID, res
end
  
# When using GANs checkpoints, read files used in each
# dataset split to resume training
function readDataSplits(metaDataPath::String)::Dict{Symbol, Vector{String}}
    fileSplit = Dict{Symbol, Vector{String}}()
    open(metaDataPath, "r") do id # open file
        content = readlines(id) # read each line
        # indices of lines to be used as reference
        trainSplit = findfirst(==("** Training:"), content)
        validateSplit = findfirst(==("** Validation:"), content)
        push!(fileSplit, :train => content[trainSplit + 2 : validateSplit - 2])
        push!(fileSplit, :validate => content[validateSplit + 2 : end])
    end
    return fileSplit
end

# read percentage of dataset used from txt metadata file
# to continue training from checkpoint
function readPercentage(metaDataName::String)::Float32
  percentage = 0f0
  open(metaDataName, "r") do id # open file
      content = vcat((readlines(id) .|> split)...)
      for index in keys(content)
        if content[index : index + 2] == ["PERCENTAGE", "OF", "DATASET:"]
          percentage = parse(Float32, content[index + 3][1 : end - 1])
          break
        end
      end
  end
  return percentage/100f0
end

# read file from topologyGAN dataset
  function readTopologyGANdataset(path; print = false)
    dictionary = Dict{Symbol, Array{Float32}}()
    h5open(path, "r") do id # open file
      dataFields = HDF5.get_datasets(id) # get references to data
      for data in dataFields
        # include each field's name and contents in dictionary
        push!(dictionary, Symbol(HDF5.name(data)[2:end]) => Float32.(HDF5.read(data)))
      end
    end
    if print
      for (key, value) in dictionary
        println(key)
        value |> statsum
      end
    end
    return dictionary
end

