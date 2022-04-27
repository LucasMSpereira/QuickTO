# Functions that involve opening/closing/saving files

using Ferrite, Parameters, HDF5, LinearAlgebra, Glob

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

# Find subgroup of dataset that meets certain criterion (e.g. plasticity, VF range etc)
function filterDataset(func, id)
  files = glob("*", "C:/Users/LucasKaoid/Desktop/datasets/data/$id") # get list of file names
  nSamples = numSample(files) # amount of samples in folder
  ##### custom hdf5 file for current analysis ###
    resultsFile = h5open("C:/Users/LucasKaoid/Desktop/datasets/data/$(rand(0:99999))", "w")
    create_dataset(resultsFile, "datasetSection", zeros(Int, nSamples))
    create_dataset(resultsFile, "sampleID", zeros(Int, nSamples))
  #####
  count = 0
  # loop in files
  @time for file in keys(files)
    forces, supps, vf, disp, top = getDataFSVDT(files[file]) # get data from file
    dataset, section = getIDs(files[file]) # dataset ID and section strings
    # loop in samples of current file
    for sample in 1:length(vf)
      count += 1
      # apply function to each sample and save alongside ID in "results" vector
      resultsFile["datasetSection"][count] = parse(Int,dataset*section)
      resultsFile["sampleID"][count] = sample
      res = func(forces[:,:,sample], supps[:,:,sample], vf[sample], disp[:,:,2*sample-1 : 2*sample], top[:,:,sample])
      if count == 1
        create_dataset(resultsFile, "result", Array{typeof(res)}(undef, nSamples))
        resultsFile["result"][1] = res
      else
        resultsFile["result"][count] = res
      end
      println("$dataset $section $sample/$(length(vf))        $count/$nSamples               $( round(Int, count/nSamples*100) )%")
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

# Returns total number of samples across files in list
numSample(files) = sum([parse(Int, split(files[g][findlast(x->x=='\\', files[g])+1:end])[3]) for g in keys(files)])

# create figure to vizualize samples
function plotSamples(FEAparams)
  # Vector of strings with paths to each dataset file
  files = glob("*", "C:/Users/LucasKaoid/Desktop/datasets/data")
  lines = FEAparams.meshSize[2]
  quantForces = 2
  colSize = 500
  count = 0
  for file in keys(files)
  # open file and read data to be plotted
    forces, supps, vf, disp, top = getDataFSVDT(files[file])
    file == 1 && (FEAparams.problems[1] = rebuildProblem(vf[1], supps[:,:,1], forces[:,:,1]))
    dataset, section = getIDs(files[file])
    mkpath("C:/Users/LucasKaoid/Desktop/datasets/fotos/$dataset")
    for i in 1:size(top,3)
      # only generate images for a fraction of samples
      rand() > 0.01 && continue
      count += 1
      vm = calcVM(prod(FEAparams.meshSize), FEAparams, disp[:,:,(2*i-1):(2*i)], 210e3*vf[i], 0.3)
      println("image $count            $( round(Int, file/length(files)*100) )%")
      # create makie figure and set it up
      fig = Figure(resolution = (1400, 700));
      colsize!(fig.layout, 1, Fixed(colSize))
      # labels for first line of grid
      Label(fig[1, 1], "Supports", textsize = 20)
      Label(fig[1, 2], "Force positions", textsize = 20)
      colsize!(fig.layout, 2, Fixed(colSize))
      supports = zeros(FEAparams.meshSize)'
      if supps[:,:,i][1,3] > 3


        if supps[:,:,i][1,3] == 4
          # left
          supports[:,1] .= 3
        elseif supps[:,:,i][1,3] == 5
          # bottom
          supports[end,:] .= 3
        elseif supps[:,:,i][1,3] == 6
          # right
          supports[:,end] .= 3
        elseif supps[:,:,i][1,3] == 7
          # top
          supports[1,:] .= 3
        end


      else

        [supports[supps[:,:,i][m, 1], supps[:,:,i][m, 2]] = 3 for m in 1:size(supps[:,:,i])[1]]

      end
      # plot supports
      heatmap(fig[2,1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,supports')
      # plot forces
      loadXcoord = zeros(quantForces)
      loadYcoord = zeros(quantForces)
      for l in 1:quantForces
        loadXcoord[l] = forces[:,:,i][l,2]
        loadYcoord[l] = lines - forces[:,:,i][l,1] + 1
      end
      # norm of weakest force. will be used to scale force vectors in arrows!() command
      fmin = 0.1*minimum(sqrt.((forces[:,:,i][:,3]).^2+(forces[:,:,i][:,4]).^2))
      axis = Axis(fig[2,2])
      xlims!(axis, -round(0.1*FEAparams.meshSize[1]), round(1.1*FEAparams.meshSize[1]))
      ylims!(axis, -round(0.1*FEAparams.meshSize[2]), round(1.1*FEAparams.meshSize[2]))
      arrows!(
        axis, loadXcoord, loadYcoord,
        forces[:,:,i][:,3], forces[:,:,i][:,4];
        lengthscale = 1/fmin
      )
      # text with values of force components
      f1 = "Forces (N):\n1: $(round(Int,forces[:,:,i][1, 3])); $(round(Int,forces[:,:,i][1, 4]))\n"
      f2 = "2: $(round(Int,forces[:,:,i][2, 3])); $(round(Int,forces[:,:,i][2, 4]))"
      l = text(
          fig[2,3],
          f1*f2;
      )

      #
          l.axis.attributes.xgridvisible = false
          l.axis.attributes.ygridvisible = false
          l.axis.attributes.rightspinevisible = false
          l.axis.attributes.leftspinevisible = false
          l.axis.attributes.topspinevisible = false
          l.axis.attributes.bottomspinevisible = false
          l.axis.attributes.xticksvisible = false
          l.axis.attributes.yticksvisible = false
          l.axis.attributes.xlabelvisible = false
          l.axis.attributes.ylabelvisible = false
          l.axis.attributes.titlevisible  = false
          l.axis.attributes.xticklabelsvisible = false
          l.axis.attributes.yticklabelsvisible = false
          l.axis.attributes.tellheight = false
          l.axis.attributes.tellwidth = false
          l.axis.attributes.halign = :center
          l.axis.attributes.width = 300
      #
      
      # labels for second line of grid
      Label(fig[3, 1], "Topology VF = $(round(vf[i];digits=3))", textsize = 20)
      Label(fig[3, 2], "von Mises (MPa)", textsize = 20)
      # plot final topology
      heatmap(fig[4, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,top[:,:,i]')
      # plot von Mises
      _,hm = heatmap(fig[4, 2],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,vm')
      # setup colorbar for von Mises
      bigVal = ceil(maximum(vm))
      t = floor(0.2*bigVal)
      t == 0 && (t = 1)
      Colorbar(fig[4, 3], hm, ticks = 0:t:bigVal)
      # save image file
      save("C:/Users/LucasKaoid/Desktop/datasets/fotos/$dataset/$(section * " " * string(i)).png", fig)
    end
  end
  println()
end

# remove plastic samples from dataset
function remPlast(id)
  files = glob("*", "C:/Users/LucasKaoid/Desktop/datasets/data/$id") # get list of file names
  # get plastic samples info
  plastic = h5open("C:/Users/LucasKaoid/Desktop/datasets/post/plastification/plastic", "r")
  pDSsec = read(plastic["datasetSection"])
  pID = read(plastic["sampleID"])
  close(plastic)
  count = 0
  # loop in files
  @time for file in keys(files)
    stringDSsection = getIDs(files[file]) # check if current file is in list of plastic samples
    if parse(Int, prod(stringDSsection)) in pDSsec
      # positions in "plastic" file that refer to samples of the current dataset
      plasticPos = findall(x -> x == parse(Int, prod(stringDSsection)), pDSsec)
      plasticQuant = length(plasticPos)
      force, supps, vf, disp, topo = getDataFSVDT(files[file]) # get data from current file
      newQuant = size(topo,3) - plasticQuant # quantity of samples in new file
      # create new file
      new = h5open("C:/Users/LucasKaoid/Desktop/datasets/data/$id/$(stringDSsection[1]) $(stringDSsection[2]) $newQuant", "w")
      initializer = zeros(size(topo,1), size(topo,2), newQuant) # common shape of sample data
      # initialize fields inside new file
      create_group(new, "inputs")
      create_dataset(new, "topologies", initializer)
      create_dataset(new["inputs"], "VF", zeros(newQuant))
      create_dataset(new["inputs"], "dispBoundConds", zeros(Int, (3,3,newQuant)))
      create_dataset(new["inputs"], "forces", zeros(2,4,newQuant))
      create_dataset(new, "disp", zeros(size(disp,1), size(disp,2), 2*newQuant))
      # IDs of samples that will be copied (elastic samples)
      a = filter!(x -> x > 0, [in(i, pID[plasticPos]) ? 0 : i for i in 1:size(topo,3)])
      # Copy elastic data to new file
      for f in 1:newQuant
        new["topologies"][:,:,f] = topo[:,:,a[f]]
        new["inputs"]["VF"][f] = vf[a[f]]
        new["inputs"]["dispBoundConds"][:,:,f] = supps[:,:,a[f]]
        new["inputs"]["forces"][:,:,f] = force[:,:,a[f]]
        new["disp"][:,:,2*f-1] = disp[:,:,2*a[f]-1]
        new["disp"][:,:,2*f] = disp[:,:,2*a[f]]
      end
      println("$(stringDSsection[1]) $(stringDSsection[2])        $file/$(length(files))       $( round(Int, file/length(files)*100) )%")
      close(new)
    else
      println("SKIPPED $(stringDSsection[1]) $(stringDSsection[2])")
      count += 1
    end
  end
  println(count)
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