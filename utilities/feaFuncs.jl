# Functions related to the FEA problem

# determine stress-strain relationship dee according to 2D stress type
function deeMat(state, e, v)
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
  return dee
end

# Generate nodeIDs used to position point loads
# However, original article "applied loads and supports to elements", not nodes
function loadPos(nels, dispBC, FEAparams, grid)
  # Random ID(s) to choose element(s) to be loaded
  global loadElements = randDiffInt(2, nels)
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
  randLoads = (-ones(length(loadElements),2) + 2*rand(length(loadElements),2))*90
  # Build matrix with positions and components of forces
  forces = [
    loadPoss[1][1] loadPoss[1][2] randLoads[1,1] randLoads[1,2]
    loadPoss[2][1] loadPoss[2][2] randLoads[2,1] randLoads[2,2]
  ]
  # Get vector with IDs of loaded nodes
  myCells = [grid.cells[g].nodes for g in loadElements]
  pos = reshape([myCells[ele][eleNode] for eleNode in 1:length(myCells[1]), ele in keys(loadElements)], (:,1))
  return pos, forces, randLoads
end

function mshData(meshSize)
  
  # Create vector of (float, float) tuples with node coordinates for "node_coords"
  # Supposes rectangular elements with unit sides staring at postition (0.0, 0.0)
  # meshSize = (x, y) = quantity of elements in each direction
  
  coordinates = Array{Tuple{Float64, Float64}}(undef, (meshSize[1]+1)*(meshSize[2]+1))
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

function mshDataQuadratic(meshSize)
  
  # Equivalent of mshData for quadratic quadrilaterals

  # Create vector of (float, float) tuples with node coordinates for "node_coords"
  # Supposes rectangular elements with unit sides staring at postition (0.0, 0.0)
  # meshSize = (x, y) = quantity of elements in each direction
  coordinates = Array{Tuple{Float64, Float64}}(undef, (2*meshSize[2]+1)*(2*meshSize[1]+1))
  # number of node columns
  ndx = 2*meshSize[2]+1
  for ID in keys(coordinates)
    # node line of current node
    nodeLine = ceil(ID/ndx)
    # node column of current node
    nodeCol = ID - floor((ID-1)/ndx)*ndx
    # y coordinate of node
    nodeY = 0.5 * (nodeLine-1)
    # x coordinate of node
    nodeX = 0.5 * (nodeCol-1)
    coordinates[ID] = (nodeX, nodeY)
  end
  
  # Create vector of tuples of integers for "cells"
  # Each line refers to a cell/element and lists its nodes in counter-clockwise order
  
  g_num = Array{Tuple{Vararg{Int, 9}}, 1}(undef, prod(meshSize))
  for elem in 1:prod(meshSize)
    # line of current element
    eleLine = ceil(elem/meshSize[1])
    # column of current element
    eleCol = elem - floor((elem-1)/meshSize[1])*meshSize[1]
    # ID of first node of current element
    nFirst = 2*(eleLine-1)*ndx+1 + (2*(eleCol-1))
    g_num[elem] = (nFirst, nFirst+2, nFirst+2*ndx+2, nFirst+2*ndx, nFirst+1, nFirst+2+ndx, nFirst+1+2*ndx, nFirst+ndx, nFirst+ndx+1)
  end
  
  return coordinates, g_num
  
end

# Run FEA with forces predicted by ML model to obtain new displacement field
function predFEA(predForce, vf, supp)
  # bound Fᵢ and Fⱼ predictions
  a, b, c, d = [], [], [], []
  @ignore_derivatives a = replace(x -> min(FEAparams.meshSize[2], x), predForce[1]) # i upper bound
  @ignore_derivatives b = replace(x -> max(1, x), a) # i lower bound
  @ignore_derivatives c = replace(x -> min(FEAparams.meshSize[1], x), predForce[2]) # j upper bound
  @ignore_derivatives d = replace(x -> max(1, x), c) # j lower bound
  predForce2 = Float64[]
  @ignore_derivatives predForce2 = [b;;d;;predForce[3];;predForce[4]] # reshape predicted forces
  # load prediction -> problem definition -> FEA -> new displacements
  solver = []
  @ignore_derivatives solver = FEASolver(
      Direct,
      rebuildProblem(convert.(Float64, cpu(vf)), convert.(Float64, cpu(supp)), convert.(Float64, cpu(predForce2)));
      xmin = 1e-6, penalty = TopOpt.PowerPenalty(3.0)
  ) # FEA solver
  feaDisp = Displacement(solver)(fill(vf, FEAparams.nElements))
  # reshape result from FEA to [nodesY, nodesX, 2]
  xDisp, yDisp = [], []
  @ignore_derivatives xDisp = quad(FEAparams.meshSize .+ 1..., [feaDisp[i] for i in 1:2:length(feaDisp)])
  @ignore_derivatives yDisp = quad(FEAparams.meshSize .+ 1..., [feaDisp[i] for i in 2:2:length(feaDisp)])
  return solidify(xDisp, yDisp)
end

# Create randomized FEA problem
function problem!(FEAparams)
  elType = "CPS4"
  if elType == "CPS4"
    grid = generate_grid(Quadrilateral, FEAparams.meshSize)
  elseif elType == "CPS8"
    grid = generate_grid(QuadraticQuadrilateral, FEAparams.meshSize)
  end
  numCellNodes = length(grid.cells[1].nodes) # number of nodes per cell/element
  nels = prod(FEAparams.meshSize) # number of elements in the mesh
  # nodeCoords = Vector of tuples with node coordinates
  # cells = Vector of tuples of integers. Each line refers to an element
  # and lists the IDs of its nodes
  if elType == "CPS4"
    nodeCoords, cells = mshData(FEAparams.meshSize)
  elseif elType == "CPS8"
    nodeCoords, cells = mshDataQuadratic(FEAparams.meshSize)
  end
  # Similar to nodeSets, but refers to groups of cells (FEA elements) 
  cellSets = Dict(
    "SolidMaterialSolid" => FEAparams.elementIDarray,
    "Eall"               => FEAparams.elementIDarray,
    "Evolumes"           => FEAparams.elementIDarray
  )
  dispBC = zeros(Int, (3,3))
  # nodeSets = dictionary mapping strings to vectors of integers. The vector groups 
  # node IDs that can be later referenced by the name in the string
  if rand() > 0.6
      # "clamp" a side
      nodeSets, dispBC = simplePins!("rand", dispBC, FEAparams)
  else
      # position pins randomly
      nodeSets, dispBC = randPins!(nels, FEAparams, dispBC, grid)
  end
  # lpos has the IDs of the loaded nodes.
  # each line in "forces" contains [forceLine forceCol forceXcomponent forceYcomponent]
  lpos, forces = loadPos(nels, dispBC, FEAparams, grid)
  # Dictionary mapping integers to vectors of floats. The vector
  # represents a force applied to the node with
  # the respective integer ID.
  cLoads = Dict(lpos[1] => forces[1,3:4])
  [merge!(cLoads, Dict(lpos[c] => forces[1,3:4])) for c in 2:numCellNodes];
  if length(lpos) > numCellNodes+1
      for pos in (numCellNodes+1):length(lpos)
          pos == (numCellNodes+1) && (global ll = 2)
          merge!(cLoads, Dict(lpos[pos] => forces[ll,3:4]))
          pos % numCellNodes == 0 && (global ll += 1)
      end
  end
  FEAparams.problems[1] = InpStiffness(
      InpContent(
          nodeCoords, elType, cells, nodeSets, cellSets,  FEAparams.V[1]*210e3, 0.3,
          0.0, Dict("supps" => [(1, 0.0), (2, 0.0)]), cLoads,
          Dict("uselessFaces" => [(1,1)]), Dict("uselessFaces" => 0.0)
      )
  )
  return FEAparams
end

# Pin a few random elements
function randPins!(nels, FEAparams, dispBC, grid)
  # generate random element IDs
  randEl = randDiffInt(3, nels)
  # get "matrix position (i,j)" of elements chosen
  suppPos = findall(x->in(x,randEl), FEAparams.elementIDmatrix)
  # build compact dispBC with pin positions chosen
  for pin in 1:length(unique(randEl))
    dispBC[pin,1] = suppPos[pin][1]
    dispBC[pin,2] = suppPos[pin][2]
    dispBC[pin,3] = 3
  end
  # get node positions of pins
  myCells = [grid.cells[g].nodes for g in randEl]
  pos = vec(reshape([myCells[ele][eleNode] for eleNode in 1:length(myCells[1]), ele in keys(randEl)], (:,1)))
  nodeSets = Dict("supps" => pos)
  return nodeSets, dispBC
end

# Rebuild FEA problem
function rebuildProblem(vf, BCs, forces)
  elementIDarray = [i for i in 1:FEAparams.nElements] # Vector that lists element IDs
  nodeCoords, cells = mshData(FEAparams.meshSize) # node coordinates and element definitions
  cellSets = Dict("SolidMaterialSolid" => elementIDarray, # associate certain properties to all elements
                  "Eall"               => elementIDarray,
                  "Evolumes"           => elementIDarray)
  grid = generate_grid(Quadrilateral, FEAparams.meshSize)
  numCellNodes = 4 # quantity of nodes per element (QUAD4)
  # matrix with element IDs in their respective position in the mesh
  elementIDmatrix = convert.(Int, quad(FEAparams.meshSize...,[i for i in 1:FEAparams.nElements]))
  # [forces[1,1:2] = ij -> elementIDmatrix -> elementID -> grid -> nodeIDs] = lpos
  # place loads
  eID = [elementIDmatrix[floor(Int, forces[f, 1]), floor(Int, forces[f, 2])] for f in 1:2]
  lpos = collect(Iterators.flatten([[grid.cells[eID[f]].nodes[e] for e in 1:4] for f in 1:2]))
  cLoads = Dict(lpos[1] => forces[1, 3:4])
  [merge!(cLoads, Dict(lpos[c] => forces[1, 3:4])) for c in 2:numCellNodes];
  if length(lpos) > numCellNodes + 1
    ll = 0
      for pos in (numCellNodes + 1) : length(lpos)
          pos == (numCellNodes + 1) && (ll = 2)
          merge!(cLoads, Dict(lpos[pos] => forces[ll, 3:4]))
          pos % numCellNodes == 0 && (ll += 1)
      end
  end
  # define nodeSets according to boundary condition (BC) definition
  if BCs[1, 3] == 4
    firstCol = [(n - 1)*(FEAparams.meshSize[1] + 1) + 1 for n in 1:(FEAparams.meshSize[2] + 1)]
    secondCol = firstCol .+ 1
    nodeSets = Dict("supps" => vcat(firstCol, secondCol))
  elseif BCs[1, 3] == 6
    firstCol = [(FEAparams.meshSize[1] + 1) * n for n in 1:(FEAparams.meshSize[2] + 1)]
    secondCol = firstCol .- 1
    nodeSets = Dict("supps" => vcat(firstCol, secondCol))
  elseif BCs[1, 3] == 5
    nodeSets = Dict("supps" => [n for n in 1:(FEAparams.meshSize[1]+1)*2])
  elseif BCs[1,3] == 7
    nodeSets = Dict("supps" => [n for n in ((FEAparams.meshSize[1]+1)*(FEAparams.meshSize[2]-1)+1):((FEAparams.meshSize[1]+1)*((FEAparams.meshSize[2]+1)))])
  else
    supports = zeros(FEAparams.meshSize)'
    [supports[BCs[m, 1], BCs[m, 2]] = 3 for m in axes(BCs)[1]]
    # supports!=0 -> elementIDmatrix -> eID -> grid -> nodes
    eID = elementIDmatrix[findall(x -> x != 0, supports)]
    nodeSets = Dict("supps" => collect(Iterators.flatten([[grid.cells[eID[s]].nodes[e] for e in 1:4] for s in axes(eID)[1]])))
  end
  return InpStiffness(InpContent(nodeCoords, "CPS4", cells, nodeSets, cellSets, vf*210e3, 0.3,
            0.0, Dict("supps" => [(1, 0.0), (2, 0.0)]), cLoads,
            Dict("uselessFaces" => [(1,1)]), Dict("uselessFaces" => 0.0)))
end

# Create the node set necessary for specific and well defined support conditions
function simplePins!(type, dispBC, FEAparams)
  type == "rand" && (type = rand(["left" "right" "top" "bottom"]))
  if type == "left" # Clamp left boundary of rectangular domain.
    fill!(dispBC, 4)
    # clamped nodes
    firstCol = [(n-1)*(FEAparams.meshSize[1]+1) + 1 for n in 1:(FEAparams.meshSize[2]+1)]
    secondCol = firstCol .+ 1
    nodeSet = Dict("supps" => vcat(firstCol, secondCol))
  elseif type == "right" # Clamp right boundary of rectangular domain.
    fill!(dispBC, 6)
    # clamped nodes
    firstCol = [(FEAparams.meshSize[1]+1)*n for n in 1:(FEAparams.meshSize[2]+1)]
    secondCol = firstCol .- 1
    nodeSet = Dict("supps" => vcat(firstCol, secondCol))
  elseif type == "bottom" # Clamp bottom boundary of rectangular domain
    fill!(dispBC, 5)
    # clamped nodes
    nodeSet = Dict("supps" => [n for n in 1:(FEAparams.meshSize[1]+1)*2])
  elseif type == "top" # Clamp top boundary of rectangular domain
    fill!(dispBC, 7)
    # clamped nodes
    # (first node of second highest line of nodes) : nodeQuant
    # (ndx)*(ndy-2)+1 : (ndx)*(ndy)
    nodeSet = Dict("supps" => [n for n in ((FEAparams.meshSize[1]+1)*(FEAparams.meshSize[2]-1)+1):((FEAparams.meshSize[1]+1)*((FEAparams.meshSize[2]+1)))])
  end
  return nodeSet, dispBC
end


# Analogous of simplePins2! but for quadratic quadrilaterals
function simplePinsQuad!(type, dispBC, FEAparams)
  type == "rand" && (type = rand(["left" "right" "top" "bottom"]))
  if type == "left"
    # Clamp left boundary of rectangular domain.
    fill!(dispBC, 4)
    # clamped nodes
    firstCol = [(n-1)*(2*FEAparams.meshSize[1]+1) + 1 for n in 1:(2*FEAparams.meshSize[2]+1)]
    secondCol = firstCol .+ 1
    nodeSet = Dict("supps" => vcat(firstCol, secondCol))
  elseif type == "right"
    # Clamp right boundary of rectangular domain.
    fill!(dispBC, 6)
    # clamped nodes
    firstCol = [(2*FEAparams.meshSize[1]+1)*n for n in 1:(2*FEAparams.meshSize[2]+1)]
    secondCol = firstCol .- 1
    nodeSet = Dict("supps" => vcat(firstCol, secondCol))
  elseif type == "bottom"
    # Clamp bottom boundary of rectangular domain
    fill!(dispBC, 5)
    # clamped nodes
    nodeSet = Dict("supps" => [n for n in 1:(2*FEAparams.meshSize[1]+1)*2])
  elseif type == "top"
    # Clamp top boundary of rectangular domain
    fill!(dispBC, 7)
    # clamped nodes
    # (first node of second highest line of nodes) : nodeQuant
    # (ndx)*(ndy-2)+1 : (ndx)*(ndy)
    nodeSet = Dict("supps" => [n for n in ((2*FEAparams.meshSize[1]+1)*(2*FEAparams.meshSize[1]-1)+1):((2*FEAparams.meshSize[1]+1)*(2*FEAparams.meshSize[2]+1))])
  end
  return nodeSet, dispBC
end

# FEA analysis of final topology
function topoFEA(forces, supps, vf, top)
  problem = rebuildProblem(vf, supps, forces) # rebuild original FEA problem from sample
  solver = FEASolver(Direct, problem; xmin=1e-6, penalty=TopOpt.PowerPenalty(3.0)) # build solver
  comp = TopOpt.Compliance(problem, solver) # compliance
  filter = DensityFilter(solver; rmin=3.0) # filtering to avoid checkerboard
  obj = x -> comp(filter(x)); # objective
  dens = [top[findfirst(x -> x == ele, FEAparams.elementIDmatrix)] for ele in 1:FEAparams.nElements]
  objVal = obj(dens)
  disp = solver.u
  dispX = []
  dispY = []
  for dof in keys(disp)
    if isodd(dof)
      dispX = vcat(dispX, disp[dof])
    else
      dispY = vcat(dispY, disp[dof])
    end
  end
  return maximum(sqrt.(dispX.^2 + dispY.^2))
end

function topologyCompliance(
  vf::T, supp::Array{Int64, 2}, force::Array{T, 2}, savedTopology::Array{T, 2}
)::Float64 where T<:Real
  problem = rebuildProblem(vf, supp, force) # InpContent struct from original problem
  solver = FEASolver(Direct, problem; xmin = 1e-6, penalty = TopOpt.PowerPenalty(3.0))
  comp = TopOpt.Compliance(solver) # define compliance
  # use comp function in final topology and return result
  comp(cat((eachslice(savedTopology; dims = 1) |> collect |> reverse)...; dims = 1))
end

# volume fraction of each topology in a batch
function volFrac(topologyBatch::Array{Float32, 4})
  [mean(topologyBatch[:, :, :, sample]) for sample in axes(topologyBatch, 4)]
end