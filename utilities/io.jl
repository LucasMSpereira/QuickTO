# Functions that involve opening/closing/saving files

using Ferrite, Parameters, HDF5, LinearAlgebra, Glob

# template to combine multiple hdf5 files into a new one
function combineFiles(pathRef)
  # get list of intermediate hdf5 analysis files
  files = glob("*", "C:/Users/LucasKaoid/Desktop/datasets/data/analysis")
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
  new = h5open("C:/Users/LucasKaoid/Desktop/datasets/post/"*pathRef, "w") # create new file to store everything
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
    new["result"][gg] = globalRes[gg] # sample ID
  end
  close(new) # close new file
end

# Create hdf5 file. Store data in a more efficient way
function createFile(quants, sec, runID, nelx,nely)
  # create file
  quickTOdata = h5open("C:/Users/LucasKaoid/Desktop/datasets/data/$runID $sec $quants", "w")
  # shape of most data info
  initializer = zeros(nely, nelx, quants)
  # organize data into folders/groups
  create_group(quickTOdata, "inputs")
  # initialize data in groups
  create_dataset(quickTOdata, "topologies", initializer)
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

# Access folder "id". Apply function "func" to all samples in "numFiles" HDF5 files.
# Store results in new HDF5 file referring to the analysis of folder "id"
function processDataset(func, id, numFiles)
  if numFiles == "end"
    files = glob("*", "C:/Users/LucasKaoid/Desktop/datasets/data/$id") # get list of file names
  else
    numFiles = parse(Int, numFiles)
    files = glob("*", "C:/Users/LucasKaoid/Desktop/datasets/data/$id")[1:numFiles] # get list of file names
  end
  nSamples = numSample(files) # total amount of samples
  ##### custom hdf5 file for current analysis ###
    resultsFile = h5open("C:/Users/LucasKaoid/Desktop/datasets/data/analysis/$(rand(0:99999))", "w")
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
      time = @elapsed begin
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
      timeElapsed += time
      str = [
        "$( round(Int, count/nSamples*100) )%    "
        "Time: $(round(Int, timeElapsed/60)) min"
      ]
      println(prod(str))
    end
  end
  close(resultsFile)
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

#=
"Reference files" store contextual info about certain samples (plastification, non-binary topology etc.).
These files also store the info needed to locate each of these samples in the dataset.
This function removes these samples from a folder of the dataset.
=#
function remSamples(id, pathRef)
  files = glob("*", "C:/Users/LucasKaoid/Desktop/datasets/data/$id") # get list of file names in folder "id"
  ds = []; sID = []; sec = [] # initialize some arrays
  # open and read reference file
  h5open("C:/Users/LucasKaoid/Desktop/datasets/post/"*pathRef, "r") do f
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
      new = h5open("C:/Users/LucasKaoid/Desktop/datasets/data/$id/$currentDS $currentSec $newQuant", "w")
      initializer = zeros(size(topo,1), size(topo,2), newQuant) # common shape of sample data
      # initialize fields inside new file
      create_group(new, "inputs")
      create_dataset(new, "topologies", initializer)
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
  dispInterp = Array{Real}(undef, prod(FEAparams.meshSize),2)
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(2), Lagrange{2,RefCube,ceil(Int, numCellNode/7)}())
  global el = 1
  # loop in elements
  for cell in CellIterator(FEAparams.problems[problemID].ch.dh)
    reinit!(cellValue, cell)
    # interpolate displacement (u, v) of element center based on nodal displacements.
    dispInterp[el,:] = function_value(cellValue, 1, disp[celldofs(cell)])
    global el += 1
  end
  # add to dataset
  quickTOdata["disp"][:, :, 2*problemID-1] = quad(FEAparams.meshSize...,dispInterp[:, 1])
  quickTOdata["disp"][:, :, 2*problemID] = quad(FEAparams.meshSize...,dispInterp[:, 2])
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