# Functions that involve opening/closing/saving files.
# Some of these functions were only used once

# Create hdf5 file. Store data in a more efficient way
function createFile(quants, sec, runID, nelx, nely)
  # create file
  quickTOdata = h5open(datasetPath * "data/$runID $sec $quants", "w")
  # organize data into folders/groups
  create_group(quickTOdata, "inputs")
  # initialize data in groups
  create_dataset(quickTOdata, "topologies", zeros(nely, nelx, quants))
  # volume fraction
  create_dataset(quickTOdata["inputs"], "VF", zeros(quants))
  # representation of mechanical supports
  create_dataset(quickTOdata["inputs"], "dispBoundConds", zeros(Int, (3,3,quants)))
  # location and value of forces
  create_dataset(quickTOdata["inputs"], "forces", zeros(2,4,quants))
  # norm of displacement vector interpolated in the center of each element
  create_dataset(quickTOdata, "disp", zeros(nely, nelx, 2*quants))
  # return file id to write info during dataset generation
  return quickTOdata
end

# print information in validation steps
function GANprints(epoch, metaData; earlyStopVals = 0)
  # if using earlystopping
  if typeof(metaData.trainConfig) == earlyStopTrainConfig
    # if first early-stop check
    if length(metaData.lossesVals[:genValHistory]) == metaData.trainConfig.earlyStopQuant + 1
      println("Epoch       Δ% Generator loss    Δ% Discriminator loss")
    end
    if earlyStopVals != 0 # If performing early-stopping check
      println(
        rpad(epoch, 12),
        rpad <| ("$(round(earlyStopVals[1]; digits = 2))%", 21)...,
        rpad <| ("$(round(earlyStopVals[2]; digits = 2))%", 21)...,
        timeNow()
      )
    else
      printGANvalid(metaData, epoch; training = :earlystop)
    end
  else # if using fixed number of epochs
    printGANvalid(metaData, epoch; training = :fixedEpochs)
  end
end

# generate PDF report about GANs
function GANreport(modelName, metaData)
  # create directory to store all PDFs
  if runningInColab == false # if running locally
    path = projPath * "/networks/GANplots/" * modelName
  else # if running in colab
    path = "./gdrive/MyDrive/dataset files/GAN saves" * modelName
  end
  mkpath(path)
  # create pdf with line plots of validation loss histories
  plotGANValHist(
    metaData.lossesVals,
    metaData.trainConfig.validFreq,
    path, modelName
  )
  # GANtestPlots(modelName, metaData)
  return nothing
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
function loadGANs(genPath, discPath)
  BSON.@load datasetPath * "data/checkpoints/" * genPath cpuGenerator
  BSON.@load datasetPath * "data/checkpoints/" * discPath cpuDiscriminator
  (cpuGenerator, cpuDiscriminator) .|> gpu
end

# Create hdf5 file. Store data in a more efficient way
function newHDF5(path, quants)
  newFile = h5open(path, "w") # create file
  create_group(newFile, "inputs")
  # initialize data in groups
  create_dataset(newFile, "topologies", zeros(FEAparams.meshSize[2], FEAparams.meshSize[1], quants))
  create_dataset(newFile["inputs"], "VF", zeros(quants)) # volume fraction
  # representation of mechanical supports
  create_dataset(newFile["inputs"], "dispBoundConds", zeros(Int, (3, 3, quants)))
  # location and value of forces
  create_dataset(newFile["inputs"], "forces", zeros(2, 4, quants))
  # norm of displacement vector interpolated in the center of each element
  create_dataset(newFile,
    "disp", zeros(FEAparams.meshSize[2] + 1, FEAparams.meshSize[1] + 1, 2 * quants)
  )
  return newFile # return file id to write info
end

# print validation information during GAN training
function printGANvalid(metaData, epoch; training)
  if training == :earlystop
    println(
        rpad(epoch, 12),
        rpad <| (sciNotation(metaData.lossesVals[:genValHistory][end], 4), 18)...,
        rpad <| (sciNotation(metaData.lossesVals[:discValHistory][end], 4), 19)...,
        "No early-stop check yet."
    )
  else
    println(
      rpad(epoch, 12),
      rpad <| (sciNotation(metaData.lossesVals[:genValHistory][end], 4), 18)...,
      rpad <| (sciNotation(metaData.lossesVals[:discValHistory][end], 4), 19)...,
      timeNow()
    )
  end
end

# Access folder "id". Apply function "func" to all samples in "numFiles" HDF5 files.
# Store results in new HDF5 file referring to the analysis of folder "id"
function processDataset(func, id; numFiles = "end")
  if numFiles == "end"
    files = glob("*", datasetPath*"data/$id") # get list of file names
  else
    numFiles = parse(Int, numFiles)
    files = glob("*", datasetPath*"data/$id")[1:numFiles] # get list of file names
  end
  nSamples = numSample(files) # total amount of samples
  ##### hdf5 file for current analysis ###
    resultsFile = h5open(datasetPath*"data/analysis/$(rand(0:99999))", "w")
    create_dataset(resultsFile, "dataset", zeros(Int, nSamples))
    create_dataset(resultsFile, "section", zeros(Int, nSamples))
    create_dataset(resultsFile, "sampleID", zeros(Int, nSamples))
  #####
  count = 0
  timeElapsed = 0.0
  # loop in files
  @time for file in keys(files)
    forces, supps, vf, disp, top = getDataFSVDT(files[file]) # get data from file
    dataset, section = getIDs(files[file]) # dataset and section IDs
    # loop in samples of current file
    for sample in 1:length(vf)
      timeElapsed += @elapsed begin
        count += 1
        # apply function to each sample and save alongside ID in "results" vector
        resultsFile["dataset"][count] = dataset
        resultsFile["section"][count] = section
        resultsFile["sampleID"][count] = sample
        res = func(
          forces[:,:,sample], supps[:,:,sample], vf[sample], disp[:,:,2*sample-1 : 2*sample], top[:,:,sample],
          dataset, section, sample
        )
        if count == 1
          create_dataset(resultsFile, "result", Array{typeof(res)}(undef, nSamples))
          resultsFile["result"][1] = res
        else
          resultsFile["result"][count] = res
        end
      end
      println("$( round(Int, count/nSamples*100) )%    Time: $(round(Int, timeElapsed/60)) min")
    end
  end
  close(resultsFile)
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
function readDataSplits(pathToMetaData::String)::Dict{Symbol, Vector{String}}
  fileSplit = Dict{Symbol, Vector{String}}()
  open(pathToMetaData, "r") do id # open file
    content = readlines(id) # read each line
    # indices of lines to be used as reference
    trainSplit = findfirst(==("** Training:"), content)
    validateSplit = findfirst(==("** Validation:"), content)
    push!(fileSplit, :train => content[trainSplit + 2 : validateSplit - 2])
    push!(fileSplit, :validate => content[validateSplit + 2 : end])
  end
  return fileSplit
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

#=
"Reference files" (created by the function 'combineFiles') store contextual info
about certain samples (plastification, non-binary topology etc.).
These files also store the info needed to locate each of these samples in the dataset.
This function removes these samples from a folder of the dataset.
=#
function remSamples(id, pathRef)
  files = glob("*", datasetPath*"data/$id") # get list of file names in folder "id"
  ds = []; sID = []; sec = [] # initialize some arrays
  # open and read reference file
  h5open(datasetPath*"analyses/"*pathRef, "r") do f
    data = read.(HDF5.get_datasets(f))
    ds = data[1]
    sID = data[3]
    sec = data[4]
  end
  @time for file in keys(files) # loop in files
    currentDS, currentSec = getIDs(files[file]) # get dataset and section IDs of current file
    pos = [] # vector to store positions in reference file that refer to samples of the current dataset file
    # loop in reference file to get positions refering to current dataset file
    for sample in keys(ds)
      if currentDS == ds[sample] && currentSec == sec[sample]
        pos = vcat(pos, sample)
      end
    end
    # check if reference file references current file
    if length(pos) > 0
      force, supps, vf, disp, topo = getDataFSVDT(files[file]) # get data from current file
      newQuant = size(topo,3) - length(pos) # quantity of samples in new file
      # create new file
      new = h5open(datasetPath*"data/$id/$currentDS $currentSec $newQuant", "w")
      # initialize fields inside new file
      create_group(new, "inputs")
      create_dataset(new, "topologies", zeros(size(topo,1), size(topo,2), newQuant))
      create_dataset(new["inputs"], "VF", zeros(newQuant))
      create_dataset(new["inputs"], "dispBoundConds", zeros(Int, (3,3,newQuant)))
      create_dataset(new["inputs"], "forces", zeros(2,4,newQuant))
      create_dataset(new, "disp", zeros(size(disp,1), size(disp,2), 2*newQuant))
      # IDs of samples that will be copied (i.e. are not referenced)
      keep = filter!(x -> x > 0, [in(i, sID[pos]) ? 0 : i for i in 1:size(topo,3)])
      # Copy part of data to new file
      for f in 1:newQuant
        new["topologies"][:,:,f] = topo[:,:,keep[f]]
        new["inputs"]["VF"][f] = vf[keep[f]]
        new["inputs"]["dispBoundConds"][:,:,f] = supps[:,:,keep[f]]
        new["inputs"]["forces"][:,:,f] = force[:,:,keep[f]]
        new["disp"][:,:,2*f-1] = disp[:,:,2*keep[f]-1]
        new["disp"][:,:,2*f] = disp[:,:,2*keep[f]]
      end
      println("$currentDS $currentSec      $( round(Int, file/length(files)*100) )% ")
      close(new)
    else
      println("SKIPPED $currentDS $currentSec")
    end
  end
end

# save both GAN NNs to BSON files
function saveGANs(metaData; finalSave = false)
  # transfer models to cpu
  cpuGenerator = cpu(metaData.generator)
  cpuDiscriminator = cpu(metaData.discriminator)
  # save models
  if runningInColab == false # if running locally
    BSON.@save datasetPath * "data/checkpoints/" * timeNow() * "gen.bson" cpuGenerator
    BSON.@save datasetPath * "data/checkpoints/" * timeNow() * "disc.bson" cpuDiscriminator
  else # if running in google colab
    BSON.@save datasetPath * "./gdrive/MyDrive/dataset files/GAN saves" * timeNow() * "gen.bson" cpuGenerator
    BSON.@save datasetPath * "./gdrive/MyDrive/dataset files/GAN saves" * timeNow() * "disc.bson" cpuDiscriminator
  end
  if !finalSave
    # bring models back to gpu, if training will continue
    metaData.generator = gpu(cpuGenerator)
    metaData.discriminator = gpu(cpuDiscriminator)
  end
  writeGANmetaData(metaData)
  return nothing
end

# get data ready to train stress CNN
function stressCNNdata(id, numFiles, FEAparams)
  files = glob("*", datasetPath*"data/$id") # get list of file names
  nSamples = numSample(files[1:numFiles]) # total number of samples
  @show nSamples
  # number of nodes in element
  numCellNodes = length(generate_grid(Quadrilateral, (140, 50)).cells[1].nodes)
  # loop in files of folder
  for file in 1:numFiles
    @show file
    forceFile, _, vf, dispFile, _ = getDataFSVDT(files[file]) # get data from file
    # create global force array or append data to it
    if file == 1 
      global force = copy(forceFile)
    else
      global force = cat(force, forceFile; dims = 3)
    end
    nels = prod(FEAparams.meshSize) # number of elements in mesh
    # loop in samples of current file
    for sample in axes(vf)[1]
      # get VM and principal stress fields for current sample
      sampleVM, _, samplePrincipals, _ = calcConds(
        nels, FEAparams, dispFile[:, :, 2 * sample - 1 : 2 * sample],
        1, 210e3 * vf[sample], 0.33, numCellNodes)
      if file == 1 && sample == 1
        global vm = copy(sampleVM)
        global principals = copy(samplePrincipals)
      else
        global vm = cat(vm, sampleVM; dims = 3)
        global principals = cat(principals, samplePrincipals; dims = 3)
      end
    end
  end
  # create new data file
  stressCNNdata = h5open(datasetPath*"data/stressCNNdata/$nSamples $(rand(1:99999))", "w")
  create_dataset(stressCNNdata, "vm", zeros(size(vm))) # von Mises stress
  create_dataset(stressCNNdata, "principals", zeros(size(principals))) # principal components
  create_dataset(stressCNNdata, "forces", zeros(size(force))) # location and value of forces
  # fill new file with data
  for gg in 1:size(vm, 3)
    stressCNNdata["vm"][:, :, gg] = vm[:, :, gg] # von Mises
    stressCNNdata["principals"][:, :, 2*gg-1 : 2*gg] = principals[:, :, 2*gg-1 : 2*gg] # principals
    stressCNNdata["forces"][:, :, gg] = force[:, :, gg] # forces
  end
  close(stressCNNdata) # close/save new file
end

# locate samples in dataset that have a specific simple mechanical support
# these samples whill be used as the test dataset for topologyGAN
# side clamped - 4: left, 5: bottom, 6: right, 7: top
function testDataset(suppID)
  for folder in 1:6 # loop in folders
    @show folder
    # list of files in current folder
    fileList = readdir(datasetPath * "/data/$(folder)"; join = true)
    for file in fileList # loop in files
        @show file
        dVFsuppFtop = 0
        # read data in current file
        h5open(file, "r") do ID dVFsuppFtop = ID |> HDF5.get_datasets .|> read end
        # indices of samples with this type of support (==(suppID)) or not (!=(suppID))
        indices = findall(!=(suppID), dVFsuppFtop[3][3, 3, :])
        length(indices) == 0 && continue # skip file if no samples of interest are found
        @show length(indices)
        new = 0
        try
        new = newHDF5(replace( # create new file
            file,
            (file |> split)[end] => length(indices) |> string
          ),
          length(indices)
        )
          writeToHDF5(new, # copy subset of data to new file
            solidify([dVFsuppFtop[1][:, :, 2 * s - 1 : 2 * s] for s in indices]...),
            dVFsuppFtop[2],
            [solidify([dVFsuppFtop[t][:, :, s] for s in indices]...) for t in 3:5]...
          )
        catch
          close(new)
        end
    end
  end
end

# prepare dataset to train topologyGAN
function topologyGANdataset(firstFileIndex, lastFileIndex)
  # get list of file names
  if firstFileIndex * lastFileIndex == 0
    files = [datasetPath * "data/test"]
  else
    files = readdir(datasetPath * "data/trainValidate"; join = true)[firstFileIndex:lastFileIndex]
  end
  numFiles = length(files)
  for (countFile, file) in enumerate(files) # loop in files of folder
    println(countFile/numFiles*100 |> round, "%")
    force, supp, vf, disp, topology = getDataFSVDT(file) # get data from file
    nSamples = length(vf)
    # initialize arrays
    vm = zeros(FEAparams.meshMatrixSize..., 1, nSamples); compliance = zeros(nSamples)
    topPlaceHolder = zeros(FEAparams.meshMatrixSize..., 1, nSamples)
    energy, binarySupp, Fx, Fy = similar(vm), similar(vm), similar(vm), similar(vm)
    for sample in axes(vf, 1) # loop in samples of current file
      sample % 1000 == 0 && @show sample
      # get VM and principal stress fields for current sample
      vm[:, :, 1, sample], energy[:, :, 1, sample] = calcCondsGAN(
        disp[:, :, 2 * sample - 1 : 2 * sample], 210e3 * vf[sample], 0.33
      )
      # determine sample compliance
      @suppress_err compliance[sample] = topologyCompliance(vf[sample],
        supp[:, :, sample], force[:, :, sample], topology[:, :, sample]
      )
      # transform and group data
      binarySupp[:, :, 1, sample] = suppToBinary(supp[:, :, sample])
      Fx[:, :, 1, sample], Fy[:, :, 1, sample] = forceToMat(force[:, :, sample])
    end
    vfMat = ones(FEAparams.meshMatrixSize..., 1, nSamples) .* reshape(vf, (1, 1, 1, :))
    # name of current file
    if firstFileIndex * lastFileIndex == 0
      fileName = "test"
    else
      fileName = file[findlast(==('\\'), file) + 1 : end]
    end
    # create new data file
    h5open(datasetPath * "data/trainValidate2/$fileName", "w") do new
      for (field, name) in zip( # create fields in file
        [compliance, vfMat, vm, energy, binarySupp, Fx, Fy, topPlaceHolder],
        ["compliance", "vf", "vm", "energy", "binarySupp","Fx", "Fy", "topologies"]
      )
        create_dataset(new, name, field |> size |> zeros)
      end
      for gg in axes(vf, 1) # fill new file with data
        new["compliance"][gg] = compliance[gg]
        new["vf"][:, :, 1, gg] = vfMat[:, :, 1, gg]
        new["vm"][:, :, 1, gg] = vm[:, :, 1, gg]
        new["energy"][:, :, 1, gg] = energy[:, :, 1, gg]
        new["binarySupp"][:, :, 1, gg] = binarySupp[:, :, 1, gg]
        new["Fx"][:, :, 1, gg] = Fx[:, :, 1, gg]
        new["Fy"][:, :, 1, gg] = Fy[:, :, 1, gg]
        new["topologies"][1:50, 1:140, 1, gg] = topology[:, :, gg]
      end
    end
  end
end