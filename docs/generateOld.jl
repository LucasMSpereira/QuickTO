using Makie, TopOpt, Parameters, StatProfilerHTML, Printf, HDF5
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent;
import GLMakie, Nonconvex
Nonconvex.@load NLopt
include(".\\functionsQuickTO")

# files to store results
topoID = h5open(".\\Results\\topologies.txt", "w")
volFracID = h5open(".\\Results\\volFrac.txt", "w")
dispID = h5open(".\\Results\\displacements.txt", "w")
vonMisesID = h5open(".\\Results\\vonMises.txt", "w")
energyID = h5open(".\\Results\\strainEnergy.txt", "w")
fileIDs = Dict(
    "topoID" => topoID,
    "volFracID" => volFracID,
    "dispID" => dispID,
    "vonMisesID" => vonMisesID,
    "energyID" => energyID
)

# struct with general parameters
@with_kw mutable struct FEAparameters
    quants::Int = 3 # number of problems
    problems::Any = Array{Any}(undef, quants) # store
    simps::Any = Array{Any}(undef, quants)
    V::Array{Real} = [0.3+rand()*0.6 for i in 1:quants] # volume fraction
    meshSize::Tuple{Int, Int} = (150, 40) # Size of rectangular mesh
    elementIDs::Array{Int} = [i for i in 1:prod(meshSize)] # Vector that lists nodeIDs
    fileIDs::Dict{String, IOStream} = fileIDs
end

FEAparams = FEAparameters()

function loopTO(FEAparameters)
    # Loop for each TO problem
    optimizers = Vector{Any}(undef, FEAparameters.quants)
    nels = prod(FEAparameters.meshSize)
    # nodeCoords = Vector of tuples with node coordinates
        # cells = Vector of tuples of integers. Each line refers to an element
        # and lists the IDs of its nodes
    nodeCoords, cells = mshData(FEAparameters.meshSize)
    # Similar to nodeSets, but refers to groups of cells (FEA elements) 
    cellSets = Dict(
        "SolidMaterialSolid" => FEAparameters.elementIDs,
        "Eall"               => FEAparameters.elementIDs,
        "Evolumes"           => FEAparameters.elementIDs)
    for i in 1:FEAparameters.quants
        print("problem $i - ")

        #
            # cellType = Type of element (CPS4 = linear quadrilateral)
            # Dictionary mapping strings to vectors of integers. The vector groups node IDs that can be later
                # referenced by the name in the string
            nodeSets = createSupps(nodeCoords, "left")
            # Dictionary mapping strings to vectors of tuples of Int and Float. The string contains a name. It refers to
                # a group of nodes defined in nodeSets. The tuples inform the displacement (Float) applied to a
                # a certain DOF (Int) of the nodes in that group. This is used to apply
                # Dirichlet boundary conditions.
            nodeDbcs = Dict("supps" => [(1, 0.0), (2, 0.0)])
            # Dictionary mapping integers to vectors of floats. The vector
                # represents a force applied to the node with the integer ID.
            cLoads = Dict(loadPos(nels, nodeSets["supps"]) => loadVals())
            # faceSets = Similar to nodeSets, but groups faces of cells (elements)
            # dLoads = Dictionary mapping strings to floats. The string refers to a group of cell faces
                # (element faces (or sides?)) defined in faceSets. The float is the value of a traction
                # applied to the faces inside that group.

            # Create struct with FEA input data
            inpCont = InpContent(nodeCoords, "CPS4", cells, nodeSets, cellSets, 210e3, 0.3,
            0.0, nodeDbcs, cLoads, Dict("uselessFaces" => [(1,1)]), Dict("uselessFaces" => 0.0))

            FEAparameters.problems[i] = inpCont

        #

        # Create TopOpt problem from inpCont struct
        problem = InpStiffness(inpCont)

        # FEA
        print("FEA - ")
        solver = FEASolver(Direct, problem; xmin=1e-6, penalty=TopOpt.PowerPenalty(3.0))
        solver()
        disp = solver.u

        #### write results to each file
        print("Writing initial data - ")
        writeVolFrac(FEAparameters, i)
        writeDisp(FEAparameters, i, disp)
        # writeField(FEAparameters, i, "vonMisesID", vmStress)
        noise = rand(nels)
        writeField(FEAparameters, i, "vonMisesID", noise)
        # writeField(FEAparameters, i, "energyID", energy)
        writeField(FEAparameters, i, "energyID", noise)

        # Definitions for optimizer
        comp = TopOpt.Compliance(problem, solver) # compliance
        filter = DensityFilter(solver; rmin=3.0) # filtering to avoid checkerboard
        obj = x -> comp(filter(x)); # objective
        x0 = fill(FEAparameters.V[i], nels) # starting densities (density = VF everywhere)
        volfrac = TopOpt.Volume(problem, solver)
        constr = x -> volfrac(filter(x)) - FEAparameters.V[i] # volume fraction constraint

        # Opotimizer setup
        print("Optimizer - ")
        alg = NLoptAlg(:LD_MMA)
        model = Nonconvex.Model(obj)
        Nonconvex.addvar!(model, zeros(nels),
                            ones(nels), init = x0)
        options = NLoptOptions()
        Nonconvex.add_ineq_constraint!(model, constr)
        global optimizers[i] = Nonconvex.optimize(model, alg, x0; options=options)
        println("Done")

        # Write topology to txt file
        writeField(FEAparameters, i, "topoID", optimizers[i].minimizer)

    end

    return optimizers

end

@time FEAparams.simps = loopTO(FEAparams)

[close(FEAparams.fileIDs[k]) for k in keys(FEAparams.fileIDs)];

# dispNLopt(FEAparams, 1)