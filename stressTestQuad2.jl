using Ferrite, TopOpt, Parameters, LinearAlgebra
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent;

#

  function mshData5(meshSize)
            
    # Create vector of (float, float) tuples with node coordinates for "node_coords"
    # Supposes rectangular elements with unit sides staring at postition (0.0, 0.0)
    # meshSize = (x, y) = quantity of elements in each direction

    coordinates = Array{Tuple{Float64, Float64}}(undef, (meshSize[1] + 1)*(meshSize[2] + 1))
    for line in 1:(meshSize[2] + 1)
        coordinates[(line + (line - 1)*meshSize[1]):(line*(1 + meshSize[1]))] .= [((col - 1)*5/1, (line - 1)*5/1) for col in 1:(meshSize[1] + 1)]
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

  function quad(x,y,vec; pa = 1)
      quad=zeros(y,x)
      for iel in 1:length(vec)
      # Line of current element
      i=floor(Int32,(iel-1)/x)+1
      # Column of current element
      j=iel-floor(Int32,(iel-1)/x)*x
      pa == 2 ? quad[y-i+1,j]=vec[iel] : quad[i,j]=vec[iel]
      end
      return quad'
  end

  function custom(FEAparams)

    # nodeCoords = Vector of tuples with node coordinates
    # cells = Vector of tuples of integers. Each line refers to an element
    # and lists the IDs of its nodes
    println("@ Defining FEA problem @")
    nodeCoords, cells = mshData5(FEAparams.meshSize)
    # Type of element (CPS4 = linear quadrilateral)
    cellType = "CPS4"
    # Dictionary mapping strings to vectors of integers. The vector groups node IDs that can be later
        # referenced by the name in the string
    # Clamp left boundary of rectangular domain
    # nodeSets = Dict("supps" => findall([iszero.(nodeCoords[node])[1] for node in 1:size(nodeCoords)[1]]))
    nodeSets = Dict("supps" => [1,2,3,4,5])
    # Similar to nodeSets, but refers to groups of cells (FEA elements) 
    cellSets = Dict(
        "SolidMaterialSolid" => FEAparams.elementIDs,
        "Eall"               => FEAparams.elementIDs,
        "Evolumes"           => FEAparams.elementIDs)
    # Young's modulus
    E = 1.0
    # Poisson's ratio
    ν = 0.3
    # Material's physical density
    density = 0.0
    # Dictionary mapping strings to vectors of tuples of Int and Float. The string contains a name. It refers to
        # a group of nodes defined in nodeSets. The tuples inform the displacement (Float) applied to a
        # a certain DOF (Int) of the nodes in that group. This is used to apply
        # Dirichlet boundary conditions.
    nodeDbcs = Dict("supps" => [(1, 0.0), (2, 0.0)])
    # Dictionary mapping integers to vectors of floats. The vector
        # represents a force applied to the node with the integer ID.
    cLoads = Dict(FEAparams.pos[1]  => [1, 0.0])
    # Similar to nodeSets, but groups faces of cells (elements)
    faceSets = Dict("uselessFaces" => [(1,1)])
    # Dictionary mapping strings to floats. The string refers to a group of cell faces
        # (element faces (or sides?)) defined in faceSets. The float is the value of a traction
        # applied to the faces inside that group.
    dLoads = Dict("uselessFaces" => 0.0)

    # Create struct with FEA input data
    inpCont = InpContent(nodeCoords, cellType, cells, nodeSets, cellSets, E, ν,
    density, nodeDbcs, cLoads, faceSets, dLoads);

    # Create TopOpt problem from inpCont struct
    return InpStiffness(inpCont; keep_load_cells=false);

  end

  @with_kw mutable struct FEAparameters
    quants::Int = 1 # number of problems
    results::Any = Array{Any}(undef, quants)
    simps::Any = Array{Any}(undef, quants)
    V::Array{Real} = [0.3, 0.65, 0.8] # volume fraction
    pos::Array{Int} = [63, 1515, 3131] # Indices of nodes to apply concentrated force
    meshSize::Tuple{Int, Int} = (4, 12) # Size of rectangular mesh
    elementIDs::Array{Int} = [i for i in 1:prod(meshSize)] # Vector that lists nodeIDs
    # file::IOStream = fileID
  end
  FEAparams = FEAparameters()

  problem = custom(FEAparams)

#

nels = length(problem.inp_content.cells)
solver = FEASolver(Direct, problem; xmin=1e-6, penalty = TopOpt.PowerPenalty(3.0))
solver.vars = fill(1.0, nels)
@time solver()
disp = solver.u

state = "stress"
e = problem.inp_content.E
v = problem.inp_content.ν

# define stress-strain matrix according to stress case
if state == "strain"

  dee = e*(1 - v)/((1 + v)*(1 - 2 * v))*
    [1 v/(1 - v) 0;
    v/(1 - v) 1 0;
    0 0 (1 - 2*v)/(2*(1 - v))]

elseif state == "stress"
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

# "Programming the finite element method", 5. ed, Wiley, pg 35
principals = Array{Any}(undef, nels)
stress = Array{Any}(undef, nels)
vm = Array{Any}(undef, nels)
centerDispGrad = Array{Any}(undef, nels)
cellValue = CellVectorValues(QuadratureRule{2, RefCube}(1), Lagrange{2,RefCube,1}())
el = 1

@time for cell in CellIterator(problem.ch.dh)
  reinit!(cellValue, cell)
  # obtain gradient of displacements interpolated on the center of the element
  global centerDispGrad[el] = function_symmetric_gradient(cellValue, 1, disp[celldofs(cell)])
  # use gradient components to build strain vector ([εₓ ε_y γ_xy])
  global ε = [centerDispGrad[el][1,1], centerDispGrad[el][2,2], centerDispGrad[el][1,2]+centerDispGrad[el][2,1]]
  # use constitutive model to calculate stresses in the center of current element
  global stress[el] = dee*ε
  σ = [
    stress[el][1] stress[el][3]
    stress[el][3] stress[el][2]
  ]
  # extract principal stresses
  global principals[el] = sort(eigvals(σ))
  # von Mises stress
  vm[el] = sqrt(stress[el]'*[1 -0.5 0; -0.5 1 0; 0 0 3]*stress[el])
  global el += 1
end
[println("$(vm[el]) $(vm[el+1]) $(vm[el+2]) $(vm[el+3])") for el in 1:4:nels];