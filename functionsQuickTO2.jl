using Ferrite, Parameters, HDF5, LinearAlgebra

function centerCoords(nels, problem)
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(1), Lagrange{2,RefCube,1}())
  centerPos = Array{Any}(undef, nels)
  el = 1
  # loop in elements
  for cell in CellIterator(problem.ch.dh)
      reinit!(cellValue, cell)
      centerPos[el] = spatial_coordinate(cellValue, 1, getcoordinates(cell))
  global el += 1
  end
  xCoords = [centerPos[i][1] for i in 1:nels]
  yCoords = [centerPos[i][2] for i in 1:nels]
  return xCoords, yCoords
end

# Create hdf5 file. Store data in a more efficient way
function createFile2(quants, folderName, nelx,nely)
  # create file
  quickTOdata = h5open("C:\\Users\\LucasKaoid\\Desktop\\datasets\\$folderName\\$(folderName)data2", "w")
  # shape of most data info
  initializer = zeros(nely, nelx, quants)
  # organize data into folders/groups
  create_group(quickTOdata, "conditions")
  create_group(quickTOdata, "inputs")
  # initialize data in groups
  create_dataset(quickTOdata, "topologies", initializer)
  # volume fraction
  create_dataset(quickTOdata["inputs"], "VF", zeros(quants))
  # representation of mechanical supports
  create_dataset(quickTOdata["inputs"], "dispBoundConds", zeros(Int, (3,3,quants)))
  # location and value of forces
  create_dataset(quickTOdata["inputs"], "forces", zeros(2,4,quants))
  # von Mises of each element
  create_dataset(quickTOdata["conditions"], "vonMises", initializer)
  # strain energy density of each element
  create_dataset(quickTOdata["conditions"], "energy", initializer)
  # norm of displacement vector interpolated in the center of each element
  create_dataset(quickTOdata["conditions"], "disp", initializer)
  # norm of stress vector σ_xy interpolated in the center of each element
  create_dataset(quickTOdata["conditions"], "stress_xy", initializer)
  # components of principal stress vector σᵢ interpolated in the center of each element
  create_dataset(quickTOdata["conditions"], "principalStress", zeros(nely, nelx, 2*quants))
  # return file id to write info during dataset generation
  return quickTOdata
end

# Generate nodeIDs used to position point loads
# However, original article "applied loads and supports to elements", not nodes
function loadPos2(nels, dispBC, FEAparams)
  # Random ID(s) to choose element(s) to be loaded
  loadElements = randDiffInt(2, nels)
  # Matrices to indicate position and component of load
  forces = zeros(2,4)'
  # i,j mesh positions of chosen elements
  global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
  
  # Verify if load will be applied on top of support.
  # Randomize positions again if that's the case
  while true
    if dispBC[1,3] > 3


      if dispBC[1,3] == 4
        # left
        if prod([loadPoss[i][2] != 1 for i in keys(loadPoss)])
          break
        else
          global loadElements = randDiffInt(2, nels)
          global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
        end
      elseif dispBC[1,3] == 5
        # bottom
        if prod([loadPoss[i][1] != FEAparams.meshSize[2] for i in keys(loadPoss)])
          break
        else
          global loadElements = randDiffInt(2, nels)
          global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
        end
      elseif dispBC[1,3] == 6
        # right
        if prod([loadPoss[i][2] != FEAparams.meshSize[1] for i in keys(loadPoss)])
          break
        else
          global loadElements = randDiffInt(2, nels)
          global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
        end
      elseif dispBC[1,3] == 7
        # top
        if prod([loadPoss[i][1] != 1 for i in keys(loadPoss)])
          break
        else
          global loadElements = randDiffInt(2, nels)
          global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
        end
      else
        println("\nProblem with dispBC\n")
      end


    else


      global boolPos = true
      for i in keys(loadPoss)
        global boolPos *= !in([loadPoss[i][k] for k in 1:2], [dispBC[h,1:2] for h in 1:size(dispBC)[1]])
      end
      if boolPos
        break
      else
        global loadElements = randDiffInt(2, nels)
        global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
      end


    end
  end
  # Generate point load component values
  randLoads = (-ones(length(loadElements),2) + 2*rand(length(loadElements),2))*10
  # Build matrix with positions and components of forces
  forces = [
    loadPoss[1][1] loadPoss[1][2] randLoads[1,1] randLoads[1,2]
    loadPoss[2][1] loadPoss[2][2] randLoads[2,1] randLoads[2,2]
  ]
  # Get vector with IDs of loaded nodes
  grid = generate_grid(Quadrilateral, FEAparams.meshSize)
  myCells = [grid.cells[g].nodes for g in loadElements]
  pos = reshape([myCells[ele][eleNode] for eleNode in 1:4, ele in keys(loadElements)], (:,1))
  return pos, forces, randLoads
end

function mshData(meshSize)
  
  # Create vector of (float, float) tuples with node coordinates for "node_coords"
  # Supposes rectangular elements with unit sides staring at postition (0.0, 0.0)
  # meshSize = (x, y) = quantity of elements in each direction
  
  coordinates = Array{Tuple{Float64, Float64}}(undef, (meshSize[1] + 1)*(meshSize[2] + 1))
  for line in 1:(meshSize[2] + 1)
    coordinates[(line + (line - 1)*meshSize[1]):(line*(1 + meshSize[1]))] .= [((col - 1)/1, (line - 1)/1) for col in 1:(meshSize[1] + 1)]
  end
  
  # Create vector of tuples of integers for "cells"
  # Each line refers to a cell/element and lists its nodes in counter-clockwise order
  
  g_num = Array{Tuple{Vararg{Int, 4}}, 1}(undef, prod(meshSize))
  for elem in 1:prod(meshSize)
    dd = floor(Int32, (elem - 1)/meshSize[1]) + elem
    g_num[elem] = (dd, dd + 1, dd + meshSize[1] + 2, dd + meshSize[1] + 1)
  end
  
  return coordinates, g_num
  
end

# create figure to vizualize sample
function plotSample2(sample, folderName, FEAparams)
  # open file and read data to be plotted
  id = h5open("C:\\Users\\LucasKaoid\\Desktop\\datasets\\$(folderName)\\$(folderName)data2", "r")
  global top = read(id["topologies"])
  global forces = read(id["inputs"]["forces"])
  global supps = read(id["inputs"]["dispBoundConds"])
  global vf = read(id["inputs"]["VF"])
  global vm = read(id["conditions"]["vonMises"])
  close(id)
  # print("von Mises ratios: ")
  # println(round.([maximum(vm[:,:,i]) for i in 1:sample]/250; digits=1))
  # create makie figure and set it up
  print("Generating image of samples: ")
  lines = size(top)[1]
  quantForces = size(forces)[1]
  colSize = 500
  for i in 1:sample
    # only generate images for a fraction of samples
    rand() > 0.1 && continue
    print("$i   ")
    fig = Figure(resolution = (1400, 700));
    colsize!(fig.layout, 1, Fixed(colSize))
    # display(fig)
    # labels for first line of grid
    Label(fig[1, 1], "supports", textsize = 20)
    Label(fig[1, 2], "force positions", textsize = 20)
    colsize!(fig.layout, 2, Fixed(colSize))
    # plot support(s) and force locations
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
    heatmap(fig[2, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,supports')
    global loadXcoord = zeros(quantForces)
    global loadYcoord = zeros(quantForces)
    for l in 1:quantForces
      global loadXcoord[l] = forces[:,:,i][l,2]
      global loadYcoord[l] = lines - forces[:,:,i][l,1] + 1
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
    # labels for second line of grid
    Label(fig[3, 1], "topology VF = $(round(vf[i];digits=3))", textsize = 20)
    Label(fig[3, 2], "von Mises (MPa)", textsize = 20)
    # plot final topology and von Mises, indicating final volume fraction
    heatmap(fig[4, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,top[:,:,i]')
    _,hm = heatmap(fig[4, 2],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,vm[:,:,i]')
    bigVal = round(maximum(vm[:,:,i]))
    t = round(0.2*bigVal)
    t == 0 && (t = 1)
    Colorbar(fig[4, 3], hm, ticks = 0:t:bigVal)
    # text with values of force components
    l = text(
        fig[2,3],
        "Forces (N):\n1: $(round(Int,forces[:,:,i][1, 3])); $(round(Int,forces[:,:,i][1, 4]))\n2: $(round(Int,forces[:,:,i][2, 3])); $(round(Int,forces[:,:,i][2, 4]))";
        # position = Point2f0(50.0, 50.0)
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
        # l = Label(fig[2,3], "JOOOOOOOJ\nJOOOOOJOJOJOJ"; tellheight = false, tellwidth = false, halign = :left)
    #
    
    save("C:\\Users\\LucasKaoid\\Desktop\\datasets\\$(folderName)\\fotos\\sample $i.png", fig)
  end
  println()
end



# reshape vectors with element quantity to reflect mesh size
function quad(nelx,nely,vec)
  # nelx = number of elements along x axis (number of columns in matrix)
  # nely = number of elements along y axis (number of lines in matrix)
  # vec = vector of scalars, each one associated to an element.
    # this vector is already ordered according to element IDs
  quad=zeros(nely,nelx)
  for i in 1:nely
    for j in 1:nelx
      global quad[nely-(i-1),j] = vec[(i-1)*nelx+1+(j-1)]
    end
  end
  return quad
end

function randPins!2(nels, FEAparams, dispBC)
  # generate random element IDs
  randEl = randDiffInt(3, nels)
  # get "matrix position (i,j)" of elements chosen
  suppPos = findall(x->in(x,randEl), FEAparams.elementIDmatrix)
  # build compact dispBC with pin positions chosen
  for pin in 1:length(unique(randEl))
    global dispBC[pin,1] = suppPos[pin][1]
    global dispBC[pin,2] = suppPos[pin][2]
    global dispBC[pin,3] = 3
  end
  # simulate grid to get mesh data (problem is not actually built yet in main file)
  grid = generate_grid(Quadrilateral, FEAparams.meshSize)
  myCells = [grid.cells[g].nodes for g in randEl]
  pos = vec(reshape([myCells[ele][eleNode] for eleNode in 1:4, ele in keys(randEl)], (:,1)))
  nodeSets = Dict("supps" => pos)
  return nodeSets, dispBC
end

# generate vector with n random and different integer values between 1 and val
function randDiffInt(n, val)
  global randVec = zeros(Int, n)
  randVec[1] = rand(1:val)
  for ind in 2:n
    global randVec[ind] = rand(1:val)
    while in(randVec[ind], randVec[1:ind-1])
      global randVec[ind] = rand(1:val)
    end
  end
  return randVec
end

# Create the node set necessary for specific and well defined support conditions
function simplePins!2(type, dispBC, FEAparams)
  type == "rand" && (type = rand(["left" "right" "top" "bottom"]))
  if type == "left"
    # Clamp left boundary of rectangular domain.
    fill!(dispBC, 4)
    # clamped nodes
    firstCol = [(n-1)*(FEAparams.meshSize[1]+1) + 1 for n in 1:(FEAparams.meshSize[2]+1)]
    secondCol = firstCol .+ 1
    nodeSet = Dict("supps" => vcat(firstCol, secondCol))
  elseif type == "right"
    # Clamp right boundary of rectangular domain.
    fill!(dispBC, 6)
    # clamped nodes
    firstCol = [(FEAparams.meshSize[1]+1)*n for n in 1:(FEAparams.meshSize[2]+1)]
    secondCol = firstCol .- 1
    nodeSet = Dict("supps" => vcat(firstCol, secondCol))
  elseif type == "bottom"
    # Clamp bottom boundary of rectangular domain
    fill!(dispBC, 5)
    # clamped nodes
    nodeSet = Dict("supps" => [n for n in 1:(FEAparams.meshSize[1]+1)*2])
  elseif type == "top"
    # Clamp top boundary of rectangular domain
    fill!(dispBC, 7)
    # clamped nodes
    nodeSet = Dict("supps" => [n for n in ((FEAparams.meshSize[1]+1)*(FEAparams.meshSize[2]-1)+1):((FEAparams.meshSize[1]+1)*((FEAparams.meshSize[2]+1)))])
  end
  return nodeSet, dispBC
end


# write displacements to file
function writeDisp(quickTOdata, problemID, disp, FEAparams)
  dispScalar = Array{Real}(undef, prod(FEAparams.meshSize))
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(1), Lagrange{2,RefCube,1}())
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

# write stresses and principal components to file
function writeStresses(nels, FEAparams, disp, problemID, e, v)
  # "Programming the finite element method", 5. ed, Wiley, pg 35
  state = "stress"
  principals = Array{Real}(undef, nels, 2) # principal stresses
  σ = Array{Real}(undef, nels) # stresses
  strainEnergy = zeros(FEAparams.meshSize)' # strain energy density in each element
  vm = zeros(FEAparams.meshSize)' # von Mises for each element
  centerDispGrad = Array{Real}(undef, nels, 2)
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(1), Lagrange{2,RefCube,1}())
  global el = 1
  # determine stress-strain relationship dee according to 2D stress type
  if state == "strain"
    # plane strain
    dee = e*(1 - v)/((1 + v)*(1 - 2 * v))*
      [1 v/(1 - v) 0;
      v/(1 - v) 1 0;
      0 0 (1 - 2*v)/(2*(1 - v))]
  
  elseif state == "stress"
    # plane stress
    dee = e/(1-v^2)*[
    1 v 0
    v 1 0
    0 0 (1-v)/2
    ]
  elseif state == "axisymmetric"
    
    dee = e*(1 - v)/((1 + v)*(1 - 2 * v))*
    [1 v/(1 - v) 0 v/(1 - v);
    v/(1 - v) 1 0 v/(1 - v);
    0 0 (1 - 2*v)/(2*(1 - v)) 0;
    v/(1 - v) v/(1 - v) 0 1]
  else
    println("Invalid stress state.")
  end
  # loop in elements
  for cell in CellIterator(FEAparams.problems[problemID].ch.dh)
    reinit!(cellValue, cell)
    # interpolate gradient of displacements on the center of the element
    global centerDispGrad = function_symmetric_gradient(cellValue, 1, disp[celldofs(cell)])
    # use gradient components to build strain vector ([εₓ ε_y γ_xy])
    global ε = [
      centerDispGrad[1,1]
      centerDispGrad[2,2]
      centerDispGrad[1,2]+centerDispGrad[2,1]
    ]
    # use constitutive model to calculate stresses in the center of current element
    global stress = dee*ε
    # take norm of stress vector to associate a scalar to each element
    global σ[el] = norm(stress)
    # extract principal stresses
    global principals[el,:] .= sort(eigvals([stress[1] stress[3]; stress[3] stress[2]]))
    elPos = findfirst(x->x==el,FEAparams.elementIDmatrix)
    # build matrix with (center) von Mises value for each element
    global vm[elPos] = sqrt(stress'*[1 -0.5 0; -0.5 1 0; 0 0 3]*stress)
    global strainEnergy[elPos] = (1+v)*(stress[1]^2+stress[2]^2+2*stress[3]^2)/(2*e) - v*(stress[1]+stress[2])^2/(2*e)
    global el += 1
  end
  # write data to file
  FEAparams.fileID["conditions"]["vonMises"][:,:,problemID] = vm
  FEAparams.fileID["conditions"]["stress_xy"][:, :, problemID] = quad(FEAparams.meshSize...,σ)
  FEAparams.fileID["conditions"]["principalStress"][:, :, 2*problemID-1] = quad(FEAparams.meshSize...,principals[:, 1])
  FEAparams.fileID["conditions"]["principalStress"][:, :, 2*problemID] = quad(FEAparams.meshSize...,principals[:, 2])
  FEAparams.fileID["conditions"]["energy"][:,:,problemID] = strainEnergy

end

function plotSampleTest2(sample, folderName, FEAparams)
  # open file and read data to be plotted
  id = h5open("C:\\Users\\LucasKaoid\\Desktop\\datasets\\$(folderName)\\$(folderName)data2", "r")
  global top = read(id["topologies"])
  global forces = read(id["inputs"]["forces"])
  global supps = read(id["inputs"]["dispBoundConds"])
  global vf = read(id["inputs"]["VF"])
  global vm = read(id["conditions"]["vonMises"])
  close(id)
  # print("von Mises ratios: ")
  # println(round.([maximum(vm[:,:,i]) for i in 1:sample]/250; digits=1))
  # create makie figure and set it up
  print("Generating image of samples: ")
  lines = size(top)[1]
  quantForces = size(forces)[1]
  colSize = 250
  for i in 1:sample
    # only generate images for a fraction of samples
    # rand() > 0.1 && continue
    print("$i   ")
    fig = Figure(resolution = (1400, 700));
    colsize!(fig.layout, 1, Fixed(colSize))
    # display(fig)
    # labels for first line of grid
    Label(fig[1, 1], "supports", textsize = 20)
    Label(fig[1, 2], "force positions", textsize = 20)
    colsize!(fig.layout, 2, Fixed(colSize))
    # plot support(s) and force locations
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
    heatmap(fig[2, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,supports')
    global loadXcoord = zeros(quantForces)
    global loadYcoord = zeros(quantForces)
    for l in 1:quantForces
      global loadXcoord[l] = forces[:,:,i][l,2]
      global loadYcoord[l] = lines - forces[:,:,i][l,1] + 1
    end
    # norm of weakest force. will be used to scale force vectors in arrows!() command
    fmin = 0.6*minimum(sqrt.((forces[:,:,i][:,3]).^2+(forces[:,:,i][:,4]).^2))
    axis = Axis(fig[2,2])
    xlims!(axis, -round(0.1*FEAparams.meshSize[1]), round(1.1*FEAparams.meshSize[1]))
    ylims!(axis, -round(0.1*FEAparams.meshSize[2]), round(1.1*FEAparams.meshSize[2]))
    arrows!(
      axis, loadXcoord, loadYcoord,
      forces[:,:,i][:,3], forces[:,:,i][:,4];
      lengthscale = 1/fmin
    )
    # labels for second line of grid
    Label(fig[3, 1], "topology VF = $(round(vf[i];digits=3))", textsize = 20)
    Label(fig[3, 2], "von Mises (MPa)", textsize = 20)
    # plot final topology and von Mises, indicating final volume fraction
    heatmap(fig[4, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,top[:,:,i]')
    _,hm = heatmap(fig[4, 2],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,vm[:,:,i]')
    bigVal = round(maximum(vm[:,:,i]))
    t = round(0.2*bigVal)
    t == 0 && (t = 1)
    Colorbar(fig[4, 3], hm, ticks = 0:t:bigVal)
    colsize!(fig.layout, 2, Fixed(colSize))
    # text with values of force components
    l = text(
        fig[2,3],
        "Forces (N):\n1: $(round(Int,forces[:,:,i][1, 3])); $(round(Int,forces[:,:,i][1, 4]))\n2: $(round(Int,forces[:,:,i][2, 3])); $(round(Int,forces[:,:,i][2, 4]))";
        # position = Point2f0(50.0, 50.0)
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
        # l = Label(fig[2,3], "JOOOOOOOJ\nJOOOOOJOJOJOJ"; tellheight = false, tellwidth = false, halign = :left)
    #
    
    save("C:\\Users\\LucasKaoid\\Desktop\\datasets\\$(folderName)\\fotos\\sample $i.png", fig)
  end
  println()
end

function miniPlot(vals, folderName, i, j, vf)
  fig = Figure(resolution = (900, 600));
  # display(fig)
  bigVal = round(maximum(vals))
  colSize = 500
  l = text(
        fig[1,1],
        "max: $(round(bigVal))\navg: $(round(mean(vals)))\nstd: $(round(std(vals)))\nVF: $(round(vf; digits=2))";
  )
  colsize!(fig.layout, 1, Fixed(150))

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

  _,hm = heatmap(fig[1,2],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,vals')
  colsize!(fig.layout, 2, Fixed(colSize))
  t = round(0.2*bigVal)
  t == 0 && (t = 1)
  Colorbar(fig[1, 3], hm, ticks = 0:t:bigVal)
  save("C:\\Users\\LucasKaoid\\Desktop\\datasets\\$(folderName)\\fotos\\sample $i\\$j.png", fig)
end


function loadPosTest(nels)
  # Random ID(s) to choose element(s) to be loaded
  loadElements = rand(1:nels,2)
  @show loadElements
  # Matrices to indicate position and component of load
  Fx = zeros((140,50))'
  Fy = zeros((140,50))'
  # i,j mesh positions of chosen elements
  global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
  @show loadPoss
  # Verify if load will be applied on top of support
  # Randomize positions again if that's the case
  # while sum(dispBC[loadPoss]) != 0
  #   global loadElements = rand(1:nels,2)
  #   global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
  # end
  # Generate point load component values
  randLoads = (-ones(length(loadElements),2) + 2*rand(length(loadElements),2))*1e3/6
  @show round.(Int,randLoads)
  # Insert load components in respective matrices
  [Fx[loadPoss[i]] = randLoads[i,1] for i in keys(loadPoss)];
  [Fy[loadPoss[i]] = randLoads[i,2] for i in keys(loadPoss)];
  # Get vector with IDs of loaded nodes
  grid = generate_grid(Quadrilateral, (140,50))
  myCells = [grid.cells[g].nodes for g in loadElements]
  @show myCells
  pos = reshape([myCells[ele][eleNode] for eleNode in 1:4, ele in keys(loadElements)], (:,1))
  @show pos
  return pos, Fx, Fy, randLoads
end

function writeStressesTest(nels, FEAparams, disp, problemID, e, v)
  # "Programming the finite element method", 5. ed, Wiley, pg 35
  state = "stress"
  principals = Array{Real}(undef, nels, 2) # principal stresses
  σ = Array{Real}(undef, nels) # stresses
  strainEnergy = zeros(FEAparams.meshSize)' # strain energy density in each element
  vm = zeros(FEAparams.meshSize)' # von Mises for each element
  centerDispGrad = Array{Real}(undef, nels, 2)
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(1), Lagrange{2,RefCube,1}())
  global el = 1
  # determine stress-strain relationship dee according to 2D stress type
  if state == "strain"
    # plane strain
    dee = e*(1 - v)/((1 + v)*(1 - 2 * v))*
      [1 v/(1 - v) 0;
      v/(1 - v) 1 0;
      0 0 (1 - 2*v)/(2*(1 - v))]
  
  elseif state == "stress"
    # plane stress
    dee = e/(1-v^2)*[
    1 v 0
    v 1 0
    0 0 (1-v)/2
    ]
  elseif state == "axisymmetric"
    
    dee = e*(1 - v)/((1 + v)*(1 - 2 * v))*
    [1 v/(1 - v) 0 v/(1 - v);
    v/(1 - v) 1 0 v/(1 - v);
    0 0 (1 - 2*v)/(2*(1 - v)) 0;
    v/(1 - v) v/(1 - v) 0 1]
  else
    println("Invalid stress state.")
  end
  # loop in elements
  for cell in CellIterator(FEAparams.problems[problemID].ch.dh)
    reinit!(cellValue, cell)
    # interpolate gradient of displacements on the center of the element
    global centerDispGrad = function_symmetric_gradient(cellValue, 1, disp[celldofs(cell)])
    # use gradient components to build strain vector ([εₓ ε_y γ_xy])
    global ε = [
      centerDispGrad[1,1]
      centerDispGrad[2,2]
      centerDispGrad[1,2]+centerDispGrad[2,1]
    ]
    # use constitutive model to calculate stresses in the center of current element
    global stress = dee*ε
    # take norm of stress vector to associate a scalar to each element
    global σ[el] = norm(stress)
    # extract principal stresses
    global principals[el,:] .= sort(eigvals([stress[1] stress[3]; stress[3] stress[2]]))
    elPos = findfirst(x->x==el,FEAparams.elementIDmatrix)
    # build matrix with (center) von Mises value for each element
    global vm[elPos] = sqrt(stress'*[1 -0.5 0; -0.5 1 0; 0 0 3]*stress)
    global strainEnergy[elPos] = (1+v)*(stress[1]^2+stress[2]^2+2*stress[3]^2)/(2*e) - v*(stress[1]+stress[2])^2/(2*e)
    global el += 1
  end

  return vm, quad(FEAparams.meshSize...,σ), quad(FEAparams.meshSize...,principals[:, 1]), quad(FEAparams.meshSize...,principals[:, 2]), strainEnergy

end

function testSamplePlot(folderName, FEAparams, i)
  # open file and read data to be plotted
  id = h5open("C:\\Users\\LucasKaoid\\Desktop\\datasets\\$(folderName)\\$(folderName)data2", "r")
  global top = read(id["topologies"])
  global forces = read(id["inputs"]["forces"])
  global supps = read(id["inputs"]["dispBoundConds"])
  global vf = read(id["inputs"]["VF"])
  global vm = read(id["conditions"]["vonMises"])
  close(id)
  lines = size(top)[1]
  quantForces = size(forces)[1]
  colSize = 250
    fig = Figure(resolution = (900, 700));
    colsize!(fig.layout, 1, Fixed(colSize))
    display(fig)
    # labels for first line of grid
    Label(fig[1, 1], "supports", textsize = 20)
    Label(fig[1, 2], "force positions", textsize = 20)
    colsize!(fig.layout, 2, Fixed(colSize))
    # plot support(s) and force locations
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
    heatmap(fig[2, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,supports')
    global loadXcoord = zeros(quantForces)
    global loadYcoord = zeros(quantForces)
    for l in 1:quantForces
      global loadXcoord[l] = forces[:,:,i][l,2]
      global loadYcoord[l] = lines - forces[:,:,i][l,1] + 1
    end
    # norm of weakest force. will be used to scale force vectors in arrows!() command
    fmin = 0.6*minimum(sqrt.((forces[:,:,i][:,3]).^2+(forces[:,:,i][:,4]).^2))
    axis = Axis(fig[2,2])
    xlims!(axis, -round(0.1*FEAparams.meshSize[1]), round(1.1*FEAparams.meshSize[1]))
    ylims!(axis, -round(0.1*FEAparams.meshSize[2]), round(1.1*FEAparams.meshSize[2]))
    arrows!(
      axis, loadXcoord, loadYcoord,
      forces[:,:,i][:,3], forces[:,:,i][:,4];
      lengthscale = 1/fmin
    )
    # labels for second line of grid
    Label(fig[3, 1], "topology VF = $(round(vf[i];digits=3))", textsize = 20)
    Label(fig[3, 2], "von Mises (MPa)", textsize = 20)
    # plot final topology and von Mises, indicating final volume fraction
    heatmap(fig[4, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,top[:,:,i]')
    _,hm = heatmap(fig[4, 2],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,vm[:,:,i]')
    bigVal = round(maximum(vm[:,:,i]))
    t = round(0.2*bigVal)
    t == 0 && (t = 1)
    Colorbar(fig[4, 3], hm, ticks = 0:t:bigVal)
    colsize!(fig.layout, 2, Fixed(colSize))
    # text with values of force components
    l = text(
        fig[2,3],
        "Forces (N):\n1: $(round(Int,forces[:,:,i][1, 3])); $(round(Int,forces[:,:,i][1, 4]))\n2: $(round(Int,forces[:,:,i][2, 3])); $(round(Int,forces[:,:,i][2, 4]))";
        # position = Point2f0(50.0, 50.0)
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
        # l = Label(fig[2,3], "JOOOOOOOJ\nJOOOOOJOJOJOJ"; tellheight = false, tellwidth = false, halign = :left)
    #
    
end

function valPlot(vals, FEAparams)
  fig = Figure(resolution = (800, 600));
  display(fig);
  bigVal = round(maximum(vals));
  _,hm = heatmap(fig[1,1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,vals');
  colsize!(fig.layout, 1, Fixed(500));
  t = round(0.2*bigVal);
  t == 0 && (t = 1);
  Colorbar(fig[1, 2], hm, ticks = 0:t:bigVal);
end