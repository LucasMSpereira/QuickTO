# Functions that involve opening/closing/saving files

# function used to combine file with data for FEAloss learning pipeline
function combineFEAlossData()
  fileList = readdir(datasetPath*"data/stressCNNdata/FEAloss"; join = true, sort = false)
  globalVF = zeros(1); globalDisp = zeros(51, 141, 1); globalSup = zeros(3, 3, 1)
  for file in fileList
    # open file, read data and concatenate to global arrays
    id = h5open(file, "r")
    globalVF = vcat(globalVF, read(id["vf"]))
    globalDisp = cat(globalDisp, read(id["disp"]); dims = 3)
    globalSup = cat(globalSup, read(id["sup"]); dims = 3)
    close(id)
    @show file
  end
  # discard null initial data
  globalVF = globalVF[2:end]; globalDisp = globalDisp[:, :, 2:end]; globalSup = globalSup[:, :, 2:end]
  count = length(globalVF)
  # create new file
  h5open(datasetPath*"data/stressCNNdata/FEAloss/FEAlossData", "w") do new
    # new = h5open(datasetPath*"data/stressCNNdata/FEAloss/FEAlossData", "w")
    # initialize fields in new file
    create_dataset(new, "vf", zeros(count))
    create_dataset(new, "disp", zeros(51, 141, 2*count))
    create_dataset(new, "sup", zeros(3, 3, count))
    # fill new file with data
    for i in 1:count
      new["vf"][i] = globalVF[i]
      new["sup"][:, :, i] = globalSup[:, :, i]
      new["disp"][:, :, 2*i-1 : 2*i] = globalDisp[:, :, 2*i-1 : 2*i]
    end
  end
end

# template to combine multiple dataset hdf5 files into a new one
function combineFiles(pathRef)
  # get list of intermediate hdf5 analysis files
  files = glob("*", datasetPath*"data/analysis")
  count = 0 # global sample counter
  globalDS = []; globalSec = []; globalSID = []; globalRes = []
  for file in keys(files) # loop in files
    # open current file and read data
    id = h5open(files[file], "r")
    ds = read(id["dataset"])
    sec = read(id["section"])
    sample = read(id["sampleID"])
    res = read(id["result"])
    close(id) # close current file
    IDs = 1:length(ds) # get IDs of samples of interest in current file (criterion changes with context)
    quants = length(IDs) # number of samples of interest in current file
    globalDS = vcat(globalDS, ds[IDs]) # dataset ID of samples
    globalSec = vcat(globalSec, sec[IDs]) # section ID of samples
    globalSID = vcat(globalSID, sample[IDs]) # IDs of samples
    globalRes = vcat(globalRes, res[IDs]) # results of samples
    count += quants # update global counter
    @show count
  end
  new = h5open(datasetPath*"analyses/"*pathRef, "w") # create new file to store everything
  # initialize fields in new file
  create_dataset(new, "dataset", zeros(Int, count))
  create_dataset(new, "section", zeros(Int, count))
  create_dataset(new, "sampleID", zeros(Int, count))
  create_dataset(new, "result", zeros(count))
  # fill new file with data of interest
  for gg in 1:count
    new["dataset"][gg] = globalDS[gg] # dataset ID
    new["section"][gg] = globalSec[gg] # section ID
    new["sampleID"][gg] = globalSID[gg] # sample
    new["result"][gg] = globalRes[gg] # result value of sample
  end
  close(new) # close new file
end

# Combine pdf files into one
function combinePDFs(path, finalName)
  PDFfiles = filter(x -> x[end-2:end] == "pdf", glob("*", path))
  read(`$(Poppler_jll.pdfunite()) $(PDFfiles) $(path)/$finalName.pdf`, String) # join pdfs together
  rm.(PDFfiles)
end

# combine files for stressCNN dataset
function combineStressCNNdata(path)
  files = glob("*", path)
  count = 0 # global sample counter
  # initialize "global" variables
  globalF = zeros(2, 4, 1); globalPrin = zeros(50, 140, 1); globalVM = zeros(50, 140, 1)
  for file in keys(files) # loop in files
    id = h5open(files[file], "r") # open current file and read data
    force = read(id["forces"]); prin = read(id["principals"]); vm = read(id["vm"])
    close(id) # close current file
    quants = size(force, 3) # amount of new samples
    # concatenate file data with "global" arrays
    globalF = cat(globalF, force; dims = 3)
    globalPrin = cat(globalPrin, prin; dims = 3)
    globalVM = cat(globalVM, vm; dims = 3)
    count += quants # update global counter
    @show count
  end
  # discard null initial data
  globalF = globalF[:, :, 2:end]; globalPrin = globalPrin[:, :, 2:end]; globalVM = globalVM[:, :, 2:end]
  # create new file to store everything
  new = h5open(datasetPath*"data/stressCNNdata/stressCNNdata", "w")
  # initialize fields in new file
  create_dataset(new, "forces", zeros(2, 4, count))
  create_dataset(new, "principals", zeros(50, 140, 2*count))
  create_dataset(new, "vm", zeros(50, 140, count))
  # fill new file with data
  for sample in 1:count
    new["forces"][:, :, sample] = globalF[:, :, sample]
    new["principals"][:, :, 2*sample-1 : 2*sample] = globalPrin[:, :, 2*sample-1 : 2*sample]
    new["vm"][:, :, sample] = globalVM[:, :, sample]
  end
  close(new) # close and save new file
end

# Create hdf5 file. Store data in a more efficient way
function createFile(quants, sec, runID, nelx,nely)
  # create file
  quickTOdata = h5open(datasetPath*"data/$runID $sec $quants", "w")
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

# create dataset to be used in new pipeline of load prediction through
# displacements. Dataset contains displacements, VF and support of samples.
# Initially restricted to samples with left side clamped.
function getFEAlossData(id)
  # get list of file names in folder "id"
  fileList = readdir(datasetPath*"data/$id"; join = true, sort = false)
  count = 0 # global sample counter
  # initialize "global" variables
  globalVF = zeros(1); globalDisp = zeros(51, 141, 1); globalSup = zeros(3, 3, 1)
  for file in fileList # loop in files
    _, sup, vf, disp, _ = getDataFSVDT(file) # read data
    samples = findall(x -> x == 4, sup[1, 3, :]) # find samples with desired support type
    # concatenate subset of file data with "global" arrays
    globalVF = vcat(globalVF, vf[samples])
    globalSup = cat(globalSup, sup[:, :, samples]; dims = 3)
    [globalDisp = cat(globalDisp, disp[:, :, 2*s-1 : 2*s]; dims = 3) for s in samples]
    count += length(samples) # update global counter
    println("count = $count - ", findfirst(x -> x == file, fileList), "/", length(fileList))
  end
  # discard null initial data
  globalVF = globalVF[2:end]; globalDisp = globalDisp[:, :, 2:end]; globalSup = globalSup[:, :, 2:end]
  # create new file to store everything
  new = h5open(datasetPath*"data/stressCNNdata/FEAlossData$id", "w")
  # initialize fields in new file
  create_dataset(new, "vf", zeros(count))
  create_dataset(new, "disp", zeros(51, 141, 2*count))
  create_dataset(new, "sup", zeros(3, 3, count))
  # fill new file with data
  for i in 1:count
    new["vf"][i] = globalVF[i]
    new["sup"][:, :, i] = globalSup[:, :, i]
    new["disp"][:, :, 2*i-1 : 2*i] = globalDisp[:, :, 2*i-1 : 2*i]
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

function loadFEAlossData()
  disp = []; sup = []; vf = []
  h5open(datasetPath*"data/stressCNNdata/FEAlossData", "r") do h5file # open hdf5 file
    datasets = HDF5.get_datasets(h5file) # get references to datasets
    # read displacement data (51 x 141 x nSamples)
    dispRead = HDF5.read(datasets[1])
    # reshape displacements to match Flux.jl's API
    disp = Array{Float32}(undef, (51, 141, 2, size(dispRead, 3) ÷ 2))
    [disp[:, :, :, sample] .= dispRead[:, :, 2*sample-1 : 2*sample] for sample in 1 : size(dispRead, 3)÷2]
    # read supports data (3 x 3 x nSamples)
    sup = convert.(Float32, HDF5.read(datasets[2]))
    # read vf data (nSamples vector)
    vf = convert.(Float32, HDF5.read(datasets[3]))
  end
  return disp, sup, vf
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

# get data ready to train stress CNN
function stressCNNdata(id, numFiles, FEAparams)
  files = glob("*", datasetPath*"data/$id") # get list of file names
  nSamples = numSample(files[1:numFiles]) # total number of samples
  @show nSamples
  # number of nodes in element
  numCellNodes = length(generate_grid(Quadrilateral, (140, 50)).cells[1].nodes)
  problem!(FEAparams) # add toy optimization problem to struct
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
    @showprogress 1 "File $file/$numFiles" for sample in axes(vf)[1]
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

# write displacements to file
function writeDisp(quickTOdata, problemID, disp, FEAparams, numCellNode)
  dispScalar = Array{Real}(undef, prod(FEAparams.meshSize))
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(2), Lagrange{2,RefCube,ceil(Int, numCellNode/7)}())
  global el = 1
  # loop in elements
  for cell in CellIterator(FEAparams.problems[problemID].ch.dh)
    reinit!(cellValue, cell)
    # interpolate displacement (u, v) of element center based on nodal displacements.
    # then take the norm of this center displacement to associate a scalar to each element
    dispScalar[el] = norm(function_value(cellValue, 1, disp[celldofs(cell)]))
    global el += 1
  end
  # reshape to represent mesh
  dispScalar = quad(FEAparams.meshSize..., dispScalar)
  # add to dataset
  quickTOdata["conditions"]["disp"][:,:,problemID] = dispScalar
end

# write displacements to file
function writeDispComps(quickTOdata, problemID, disp, FEAparams, numCellNode)
  dispInterp = Array{Real}(undef, prod(FEAparams.meshSize), 2)
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(2), Lagrange{2, RefCube, ceil(Int, numCellNode/7)}())
  el = 1
  for cell in CellIterator(FEAparams.problems[problemID].ch.dh) # loop in elements
    reinit!(cellValue, cell)
    # interpolate displacement (u, v) of element center based on nodal displacements.
    dispInterp[el, :] = function_value(cellValue, 1, disp[celldofs(cell)])
    el += 1
  end
  # add to dataset
  quickTOdata["disp"][:, :, 2 * problemID - 1] = quad(FEAparams.meshSize..., dispInterp[:, 1])
  quickTOdata["disp"][:, :, 2 * problemID] = quad(FEAparams.meshSize..., dispInterp[:, 2])
  return dispInterp
end

# write stresses, principal components and strain energy density to file
function writeConds(fileID, vm, σ, principals, strainEnergy, problemID, FEAparams)

  fileID["conditions"]["vonMises"][:,:,problemID] = vm
  fileID["conditions"]["stress_xy"][:, :, problemID] = quad(FEAparams.meshSize...,σ)
  fileID["conditions"]["principalStress"][:, :, 2*problemID-1] = quad(FEAparams.meshSize...,principals[:, 1])
  fileID["conditions"]["principalStress"][:, :, 2*problemID] = quad(FEAparams.meshSize...,principals[:, 2])
  fileID["conditions"]["energy"][:,:,problemID] = strainEnergy

end