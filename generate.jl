using Makie, TopOpt, Statistics
using TopOpt.TopOptProblems.Visualization: visualize
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
import GLMakie

# A few declarations
n = 3 # number of problems
result = Array{Any}(undef, n)
simps = Array{Any}(undef, n)
V = [0.5, 0.65, 0.8] # volume fraction
pos = [3, 8, 11]

#

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

# Loop for each TO problem
@time for i in 1:n

    println("\n@@ loop $i @@\n")
    #### Definition of FEA problem

    # Vector of tuples with node coordinates
    println("@ Defining FEA problem @")
    nodeCoords = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (3.0, 1.0),
        (0.0, 2.0),
        (1.0, 2.0),
        (2.0, 2.0),
        (3.0, 2.0)
    ]
    # Type of element (CPS4 = linear quadrilateral)
    cellType = "CPS4"
    # Vector of tuples. Each line refers to an element and lists the IDs of its nodes
    cells = [
        (1,2,6,5),
        (2,3,7,6),
        (3,4,8,7),
        (5,6,10,9),
        (6,7,11,10),
        (7,8,12,11)
    ]
    # Dictionary mapping strings to vectors of integers. The vector groups node IDs that can be later
    # referenced by the name in the string
    nodeSets = Dict("supps" => [1, 5, 9])
    # Similar to nodeSets, but refers to groups of cells (FEA elements) 
    cellSets = Dict(
        "SolidMaterialSolid" => [1, 2, 3, 4, 5, 6],
        "Eall"               => [1, 2, 3, 4, 5, 6],
        "Evolumes"           => [1, 2, 3, 4, 5, 6])
    # Young's modulus
    E = 210e9
    # Poisson's ratio
    ν = 0.3
    # Material's physical density
    density = 7850.0
    # Dictionary mapping strings to vectors of tuples of Int and Float. The string contains a name. It refers to
    # a group of nodes defined in nodeSets. The tuples inform the displacement (Float) applied to a
    # a certain DOF (Int) of the nodes in that group. This is used to apply
    # Dirichlet boundary conditions.
    nodeDbcs = Dict("supps" => [(1, 0.0), (2, 0.0)])
    # Dictionary mapping integers to vectors of floats. The vector
    # represents a force applied to the node with the integer ID.
    cLoads = Dict(pos[i] => [0.0, 1e3])
    # Similar to nodeSets, but groups faces of cells (elements)
    faceSets = Dict("uselessFaces" => [(1,1)])
    # Dictionary mapping strings to floats. The string refers to a group of cell faces
    # (element faces (or sides?)) defined in faceSets. The float is the value of a traction
    # applied to the faces inside that group.
    dLoads = Dict("uselessFaces" => 0.0)

    # Create struct with FEA input data
    inpCont = InpContent(nodeCoords, cellType, cells, nodeSets, cellSets, E, ν,
    density, nodeDbcs, cLoads, faceSets, dLoads)

    # Create TopOpt problem from inpCont struct
    problem = InpStiffness(inpCont; keep_load_cells=true)

    ### Topological optimization setup

    println("@ Topological optimization setup @")
    xmin = 1e-6 # minimum density
    rmin = 2.0; # density filter radius

    penalty = TopOpt.PowerPenalty(3.0)
    solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)

    comp = TopOpt.Compliance(problem, solver)
    filter = DensityFilter(solver; rmin=rmin)
    obj = x -> comp(filter(x))

    volfrac = TopOpt.Volume(problem, solver)
    constr = x -> volfrac(filter(x)) - V[i]

    ### Optimizer
    println("@ MMA setup @")
    mma_options =
        options = MMAOptions(;
            maxiter=3000, tol=Nonconvex.Tolerance(; x=1e-3, f=1e-3, kkt=0.001)
        )
    convcriteria = Nonconvex.KKTCriteria()
    x0 = fill(V[i], length(solver.vars))
    optimizer = Optimizer(
        obj, constr, x0, MMA87(); options=mma_options, convcriteria=convcriteria
    )

    simps[i] = SIMP(optimizer, solver, penalty.p);

    result[i] = simps[i](x0);

end

#
    #@show result.convstate
    #@show optimizer.workspace.iter
    #@show result.objval

    # fig = visualize(problem; topology = result[1].topology,
    #     default_exagg_scale = 0.07,
    #     scale_range = 10.0, vector_linewidth = 3, vector_arrowsize = 0.5,
    # )
    # Makie.display(fig)
#

topo = quad(3,2,result[3].topology)
Makie.heatmap(topo)