using TopOpt, Makie, Nonconvex, Parameters, Makie, Statistics, Test
using TopOpt.TopOptProblems.Visualization: visualize
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
import GLMakie
Nonconvex.NonconvexCore.show_residuals[] = true

#

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
      nodeCoords, cells = mshData(FEAparams.meshSize)
      # Type of element (CPS4 = linear quadrilateral)
      cellType = "CPS4"
      # Dictionary mapping strings to vectors of integers. The vector groups node IDs that can be later
          # referenced by the name in the string
      # Clamp left boundary of rectangular domain
      nodeSets = Dict("supps" => findall([iszero.(nodeCoords[node])[1] for node in 1:size(nodeCoords)[1]]))
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
      cLoads = Dict(FEAparams.pos[1] => [0.0, -1.0])
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

#

@with_kw mutable struct FEAparameters
  quants::Int = 1 # number of problems
  results::Any = Array{Any}(undef, quants)
  simps::Any = Array{Any}(undef, quants)
  V::Array{Real} = [0.3, 0.65, 0.8] # volume fraction
  pos::Array{Int} = [3381, 1515, 3131] # Indices of nodes to apply concentrated force
  meshSize::Tuple{Int, Int} = (160, 40) # Size of rectangular mesh
  elementIDs::Array{Int} = [i for i in 1:prod(meshSize)] # Vector that lists nodeIDs
  # file::IOStream = fileID
end
FEAparams = FEAparameters()

problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), 1.0, 0.3, 1.0)
# problem = HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), 1.0, 0.3, 1.0)
# problem = LBeam(Val{:Linear}, Float64; force=f)
# problem = TieBeam(Val{:Quadratic}, Float64)

problemCustom = custom(FEAparams)

x0 = fill(0.7, 160*40)

penalty = TopOpt.PowerPenalty(3.0)
solver = FEASolver(Direct, problem; xmin=1e-6, penalty=penalty)
solver.vars = x0
solver()
disp = solver.u

solverCustom = FEASolver(Direct, problemCustom; xmin=1e-6, penalty=penalty)
solverCustom.vars = x0
solverCustom()
dispCustom = solverCustom.u

#
  # println(fieldnames(typeof(solver)))
  # println(fieldnames(typeof(solver.qr)))
#

println(mean(disp))
println(mean(dispCustom))