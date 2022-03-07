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


# Create hdf5 file
function createFile(quants,nelx,nely)
  # create file
  quickTOdata = h5open("quickTOdata", "w")
  # shape of data info
  initializer = zeros(nely, nelx, quants)
  # organize data into folders/groups
  create_group(quickTOdata, "conditions")
  create_group(quickTOdata, "inputs")
  # initialize data in groups
  create_dataset(quickTOdata, "topologies", initializer)
  # volume fraction
  create_dataset(quickTOdata["inputs"], "VF", initializer)
  # representation of mechanical supports
  create_dataset(quickTOdata["inputs"], "dispBoundConds", initializer)
  # location and value of x components of forces
  create_dataset(quickTOdata["inputs"], "Fx", initializer)
  # location and value of y components of forces
  create_dataset(quickTOdata["inputs"], "Fy", initializer)
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


# Reshape densities vector to mirror mesh layout (quad()), then plot using Makie.heatmap()
dispDens(FEAparams, problem) = display(
  Makie.heatmap(1:FEAparams.meshSize[1],
    FEAparams.meshSize[2]:-1:1,
    quad(FEAparams.meshSize...,
      FEAparams.simps[problem].result.topology)')
)

# Same as dispDens, but acessing NLopt's solution format, instead of NonconvexMMA's
dispNLopt(FEAparams, problem) = display(
  Makie.heatmap(1:FEAparams.meshSize[1],
    FEAparams.meshSize[2]:-1:1,
    quad(FEAparams.meshSize...,
    FEAparams.simps[problem].minimizer)')
)

# Generate nodeIDs used to position point loads
# However, original article "applied loads and supports to elements", not nodes
function loadPos(nels, dispBC, FEAparams)
  # Random ID(s) to choose element(s) to be loaded
  loadElements = rand(1:nels,2)
  # Matrices to indicate position and component of load
  Fx = zeros(FEAparams.meshSize)'
  Fy = zeros(FEAparams.meshSize)'
  # i,j mesh positions of chosen elements
  global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
  # Verify if load will be applied on top of support
  # Randomize positions again if that's the case
  while sum(dispBC[loadPoss]) != 0
    global loadElements = rand(1:nels,2)
    global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
  end
  # Generate point load component values
  randLoads = (-ones(length(loadElements),2) + 2*rand(length(loadElements),2))*1e3/4
  # Insert load components in respective matrices
  [Fx[loadPoss[i]] = randLoads[i,1] for i in keys(loadPoss)];
  [Fy[loadPoss[i]] = randLoads[i,2] for i in keys(loadPoss)];
  # Get vector with IDs of loaded nodes
  grid = generate_grid(Quadrilateral, FEAparams.meshSize)
  myCells = [grid.cells[g].nodes for g in loadElements]
  pos = reshape([myCells[ele][eleNode] for eleNode in 1:4, ele in keys(loadElements)], (:,1))
  return pos, Fx, Fy, randLoads
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
function plotSample(sample)
  # open file and read data to be plotted
  id = h5open("quickTOdata", "r")
  top = read(id["topologies"])
  fx = read(id["inputs"]["Fx"])
  fy = read(id["inputs"]["Fy"])
  supps = read(id["inputs"]["dispBoundConds"])
  vf = read(id["inputs"]["VF"])
  vm = read(id["conditions"]["vonMises"])
  close(id)
  print("von Mises ratios: ")
  println(round.([maximum(vm[:,:,i]) for i in 1:FEAparams.quants]/250;digits=2))
  # create makie figure and set it up
  for i in 1:sample
    fig = Figure(resolution = (1200, 700));
    colSize = 500
    colsize!(fig.layout, 1, Fixed(colSize))
    # display(fig)
    # labels for first line of grid
    Label(fig[1, 1], "supports", textsize = 20)
    Label(fig[1, 2], "force positions", textsize = 20)
    colsize!(fig.layout, 2, Fixed(colSize))
    # plot support(s) and force locations
    heatmap(fig[2, 1],1:140,50:-1:1,supps[:,:,i]')
    loadMatPos = findall(!iszero, fx[:,:,i])
    loadXcoord = zeros(length(loadMatPos))
    loadYcoord = zeros(length(loadMatPos))
    for l in keys(loadMatPos)
      global loadXcoord[l] = loadMatPos[l][2]
      global loadYcoord[l] = size(fx)[1] - loadMatPos[l][1] + 1
    end
    # norm of weakest force. will be used to scale force vectors in arrows!() command
    fmin = 0.1*minimum(sqrt.((fx[:,:,i][loadMatPos]).^2+(fy[:,:,i][loadMatPos]).^2))
    axis = Axis(fig[2,2])
    xlims!(axis, 0, 140)
    ylims!(axis, 0, 50)
    arrows!(
      axis, loadXcoord, loadYcoord,
      fx[:,:,i][loadMatPos], fy[:,:,i][loadMatPos];
      lengthscale = 1/fmin
    )
    # labels for second line of grid
    Label(fig[3, 1], "topology VF=$(round(vf[1,1,i];digits=3))", textsize = 20)
    Label(fig[3, 2], "von Mises (MPa)", textsize = 20)
    # plot final topology and von Mises, indicating final volume fraction
    heatmap(fig[4, 1],1:140,50:-1:1,top[:,:,i]')
    _,hm = heatmap(fig[4, 2],1:140,50:-1:1,vm[:,:,i]')
    Colorbar(fig[4, 3], hm, ticks = 0:30:(maximum(vm[:,:,i])))
    save(".\\fotos\\sample $i.png", fig)
  end
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

function randSupps!(nels, FEAparams, dispBC)
  # generate random element IDs
  randEl = rand(1:nels,3)
  # alter dispBC in the support positions chosen
  dispBC[findall(x->in(x,randEl), FEAparams.elementIDmatrix)] .= 3
  # simulate grid to get mesh data (problem is not actually built yet)
  grid = generate_grid(Quadrilateral, FEAparams.meshSize)
  myCells = [grid.cells[g].nodes for g in randEl]
  pos = vec(reshape([myCells[ele][eleNode] for eleNode in 1:4, ele in keys(randEl)], (:,1)))
  nodeSets = Dict("supps" => pos)
  return nodeSets, dispBC
end

# Create the node set necessary for specific and well defined support conditions
function simpleSupps!(type, dispBC, FEAparams)
  type == "rand" && (type = rand(["left" "right" "top" "bottom"]))
  if type == "left"
    # Clamp left boundary of rectangular domain.
    # clamped elements
    elements = [(n-1)*FEAparams.meshSize[1] + 1 for n in 1:(FEAparams.meshSize[2])]
    # clamped nodes
    firstCol = [(n-1)*(FEAparams.meshSize[1]+1) + 1 for n in 1:(FEAparams.meshSize[2]+1)]
    secondCol = firstCol .+ 1
    nodeSet = Dict("supps" => vcat(firstCol, secondCol))
  elseif type == "right"
    # Clamp right boundary of rectangular domain.
    # clamped elements
    elements = [FEAparams.meshSize[1]*n for n in 1:FEAparams.meshSize[2]]
    # clamped nodes
    firstCol = [(FEAparams.meshSize[1]+1)*n for n in 1:FEAparams.meshSize[2]]
    secondCol = firstCol .- 1
    nodeSet = Dict("supps" => vcat(firstCol, secondCol))
  elseif type == "bottom"
    # Clamp bottom boundary of rectangular domain
    # clamped elements
    elements = [n for n in 1:FEAparams.meshSize[1]]
    # clamped nodes
    nodeSet = Dict("supps" => [n for n in 1:(FEAparams.meshSize[1]+1)*2])
  elseif type == "top"
    # Clamp top boundary of rectangular domain
    # clamped elements
    elements = [n for n in (FEAparams.meshSize[1]*(FEAparams.meshSize[2]-1)+1):prod(FEAparams.meshSize)]
    meshSize = (140,50)
    [n for n in (meshSize[1]*(meshSize[2]-1)+1):prod(meshSize)];
    # clamped nodes
    nodeSet = Dict("supps" => [n for n in ((FEAparams.meshSize[1]+1)*(FEAparams.meshSize[2]-1)+1):((FEAparams.meshSize[1]+1)*((FEAparams.meshSize[2]+1)))])
    [n for n in ((meshSize[1]+1)*(meshSize[2]-1)+1):((meshSize[1]+1)*((meshSize[2]+1)))];
  end
  dispBC[findall(x->in(x,elements), FEAparams.elementIDmatrix)] .= 3
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
    # interpolate displacement (u, v) of element center, based on nodal displacements
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
  FEAparams.fileID["conditions"]["vonMises"][:,:,problemID] = vm
  FEAparams.fileID["conditions"]["stress_xy"][:, :, problemID] = quad(FEAparams.meshSize...,σ)
  FEAparams.fileID["conditions"]["principalStress"][:, :, 2*problemID-1] = quad(
    FEAparams.meshSize...,
    principals[:, 1]
  )
  FEAparams.fileID["conditions"]["principalStress"][:, :, 2*problemID] = quad(
    FEAparams.meshSize...,
    principals[:, 2]
  )
  FEAparams.fileID["conditions"]["energy"][:,:,problemID] = strainEnergy

end



