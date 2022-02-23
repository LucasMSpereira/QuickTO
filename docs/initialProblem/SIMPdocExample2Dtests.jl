using TopOpt, Makie, Nonconvex, Parameters, Makie
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent;
import GLMakie

Nonconvex.NonconvexCore.show_residuals[] = true
# Nonconvex.@load NLopt


E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force

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

#

# problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f)
# problem = HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f)
# problem = LBeam(Val{:Linear}, Float64; force=f)
# problem = TieBeam(Val{:Quadratic}, Float64)

#

    @with_kw mutable struct FEAparameters
        quants::Int = 1 # number of problems
        results::Any = Array{Any}(undef, quants)
        simps::Any = Array{Any}(undef, quants)
        V::Array{Real} = [0.3, 0.65, 0.8] # volume fraction
        pos::Array{Int} = [6501, 1515, 3131] # Indices of nodes to apply concentrated force
        meshSize::Tuple{Int, Int} = (160, 40) # Size of rectangular mesh
        elementIDs::Array{Int} = [i for i in 1:prod(meshSize)] # Vector that lists nodeIDs
        # file::IOStream = fileID
    end

    FEAparams = FEAparameters()

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
    E = 210e3
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
    cLoads = Dict(FEAparams.pos[1] => [1e3, -1e3])
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
    problem = InpStiffness(inpCont; keep_load_cells=false);

#

V = 0.3 # volume fraction

xmin = 1e-6 # minimum density
rmin = 3.0 # density filter radius

penalty = TopOpt.PowerPenalty(3.0)
solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)

comp = TopOpt.Compliance(problem, solver)
filter = DensityFilter(solver; rmin=rmin)
obj = x -> comp(filter(x))

volfrac = TopOpt.Volume(problem, solver)
constr = x -> volfrac(filter(x)) - V
x0 = fill(V, length(solver.vars))

### Optimizer
println("@ MMA setup @")
mma_options =
    options = MMAOptions(;
        maxiter=3000, tol=Nonconvex.Tolerance(; x=1e-3, f=1e-3, kkt=1e-6)
    );
convcriteria = Nonconvex.KKTCriteria();
optimizer = Optimizer(
    obj, constr, x0, MMA87(); options=mma_options, convcriteria=convcriteria
);

simp = SIMP(optimizer, solver, penalty.p)

println("##########    0    ###########")
@time results = simp(x0)

println("##########    1    ###########")
typeof(simp.result.topology)
display(heatmap(quad(FEAparams.meshSize[1], FEAparams.meshSize[2],
                                    simp.result.topology)))


#=
-+Esse arquivo atualmente funcionou para o problema de PointLoadCantilever.
-+testar outros tipos de problema.
-+visualização de alguns deu problema, mas todos rodaram
-Mudar definição de problema para a minha
    personalizada (que deve representar um tipo padrão de problema em OT)
-Se etapa anterior der problema, há alguma diferença entre a minha construção
    do problema e as funções automáticas. O problema então não é parametrização
    da OT ou do otimizador.
-Adaptar para usar o otimizador em NLopt. Mohamed disse que é mais robusto.
=#

# grade = Grid(Ferrite.Quadrilateral.(inpCont.cells), Ferrite.Node.(inpCont.node_coords))
# dh = Ferrite.DofHandler(grade)
# typeof(dh)
# fieldnames(typeof(dh))
# dh.field_interpolations
# println(fieldnames(typeof(problem2)))
# println(fieldnames(typeof(problem)))
# println(fieldnames(typeof(inpCont)))
# println(fieldnames(typeof(problem2.metadata)))
# println(fieldnames(typeof(problem.ch)))
# println(problem.ch.dbcs)
# println(fieldnames(typeof(problem.ch.dh)))
# println(fieldnames(typeof(problem.ch.dh.bc_values[1])))
# size(problem.ch.dh.bc_values[1].M)
# [[println(problem.ch.dh.bc_values[1].M[j,i,:]) for i in 1:2] for j in 1:4]
# println(fieldnames(typeof(problem.ch.dh.grid)))
# problem.ch.dh.grid.cells
# println(fieldnames(typeof(problem.ch.dh.grid)))
# problem.ch.dh.grid.nodes
# println(keys(problem.ch.dh.grid.cellsets))
# println(keys(problem.ch.dh.grid.nodesets))
# println(keys(problem.ch.dh.grid.facesets))
# println(problem.ch.dh.grid.boundary_matrix)
# println(fieldnames(typeof(problem.black)))
# println(fieldnames(typeof(problem.metadata)))
# problem.metadata.cell_dofs
# println(problem.metadata.node_dofs[:,162])
# cells[1]