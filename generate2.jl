using Makie, TopOpt, Parameters, StatProfilerHTML, Printf, HDF5, Statistics
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent;
import GLMakie, Nonconvex
Nonconvex.@load NLopt;
include(".\\functionsQuickTO2.jl")

# random folder name
folderName = string(rand(0:9999))
mkpath("C:\\Users\\LucasKaoid\\Desktop\\datasets\\$(folderName)\\fotos")

# struct with general parameters
@with_kw mutable struct FEAparameters
    forces::Int = 2 # number of forces
    quants::Int = 30 # number of problems
    problems::Any = Array{Any}(undef, quants) # store FEA problem structs
    V::Array{Real} = [0.3+rand()*0.6 for i in 1:quants] # volume fraction
    meshSize::Tuple{Int, Int} = (20, 20) # Size of rectangular mesh
    elementIDarray::Array{Int} = [i for i in 1:prod(meshSize)] # Vector that lists element IDs
    # matrix with element IDs in their respective position in the mesh
    elementIDmatrix::Array{Int,2} = convert.(Int, quad(meshSize...,[i for i in 1:prod(meshSize)]))
    # hdf5 file to store data
    fileID = createFile2(quants, folderName, meshSize...)
end
FEAparams = FEAparameters()

function loopTO(FEAparameters)
    # Loop for each TO problem

    nels = prod(FEAparameters.meshSize) # number of elements in the mesh
    # nodeCoords = Vector of tuples with node coordinates
        # cells = Vector of tuples of integers. Each line refers to an element
        # and lists the IDs of its nodes
    nodeCoords, cells = mshData(FEAparameters.meshSize)
    # Similar to nodeSets, but refers to groups of cells (FEA elements) 
    cellSets = Dict(
        "SolidMaterialSolid" => FEAparameters.elementIDarray,
        "Eall"               => FEAparameters.elementIDarray,
        "Evolumes"           => FEAparameters.elementIDarray
    )
    
    tempo = zeros(FEAparameters.quants)
    for i in 1:FEAparameters.quants
        tempo[i] = @elapsed begin
        print("problem $i - ")

        # integer matrix representing displacement boundary conditions (supports):
            # 0: free element
            # 1: element restricted in the x direction ("roller")
            # 2: element restricted in the y direction ("roller")
            # 3: element restricted in both directions ("pinned"/"clamped")
        dispBC = zeros(Int, (3,3))

        #
            # nodeSets = dictionary mapping strings to vectors of integers. The vector groups 
                # node IDs that can be later referenced by the name in the string
            if rand() > 0.5
                # "clamp" a side
                nodeSets, dispBC = simplePins!2("rand", dispBC, FEAparams)
            else
                # position pins randomly
                nodeSets, dispBC = randPins!2(nels, FEAparams, dispBC)
            end
            
            # lpos has the IDs of the loaded nodes.
            # each line in "forces" contains [forceLine forceCol forceXcomponent forceYcomponent]
            lpos, forces = loadPos2(nels, dispBC, FEAparameters)
            # Dictionary mapping integers to vectors of floats. The vector
                # represents a force applied to the node with
                # the respective integer ID.
            cLoads = Dict(lpos[1] => forces[1,3:4])
            [merge!(cLoads, Dict(lpos[c] => forces[1,3:4])) for c in 2:4];
            if length(lpos) > 4
                for pos in 5:length(lpos)
                    pos == 5 && (global ll = 2)
                    merge!(cLoads, Dict(lpos[pos] => forces[ll,3:4]))
                    pos % 4 == 0 && (global ll += 1)
                end
            end
            
            # faceSets = Similar to nodeSets, but groups faces of cells (elements)
            # dLoads = Dictionary mapping strings to floats. The string refers to a group of cell faces
                # (element faces (or sides?)) defined in faceSets. The float is the value of a traction
                # applied to the faces inside that group.
            # nodeDbcs = Dictionary mapping strings to vectors of tuples of Int and Float. The
                # string contains a name. It refers to
                # a group of nodes defined in nodeSets. The tuples inform the displacement (Float) applied to a
                # a certain DOF (Int) of the nodes in that group. This is used to apply
                # Dirichlet boundary conditions.
            # cellType = Type of element (CPS4 = linear quadrilateral)
            
            # Create TopOpt problem from inpCont struct
            problem = InpStiffness(InpContent(
                nodeCoords, "CPS4", cells, nodeSets, cellSets, FEAparameters.V[i]*210e3, 0.3,
                0.0, Dict("supps" => [(1, 0.0), (2, 0.0)]), cLoads,
                Dict("uselessFaces" => [(1,1)]), Dict("uselessFaces" => 0.0)))

            FEAparameters.problems[i] = problem

        #

        # FEA
        print("FEA - ")
        solver = FEASolver(Direct, problem; xmin=1e-6, penalty=TopOpt.PowerPenalty(3.0))
        solver()
        # disp = copy(solver.u)
        disp = solver.u

        #### write data to file
        print("Writing initial data - ")
        # write volume fraction to file
        FEAparameters.fileID["inputs"]["VF"][i] = FEAparameters.V[i]
        # write displacements boundary conditions to file
        FEAparameters.fileID["inputs"]["dispBoundConds"][:,:,i] = dispBC
        # write forces to file
        FEAparameters.fileID["inputs"]["forces"][:,:,i] = forces
        # write displacements to file
        writeDisp(FEAparameters.fileID, i, disp, FEAparameters)
        # write stresses to file and verify material linearity
        writeStresses(nels, FEAparameters, disp, i, FEAparameters.V[i]*210e3, 0.3)

        # Definitions for optimizer
        comp = TopOpt.Compliance(problem, solver) # compliance
        filter = DensityFilter(solver; rmin=3.0) # filtering to avoid checkerboard
        obj = x -> comp(filter(x)); # objective
        x0 = fill(FEAparameters.V[i], nels) # starting densities (VF everywhere)
        volfrac = TopOpt.Volume(problem, solver)
        constr = x -> volfrac(filter(x)) - FEAparameters.V[i] # volume fraction constraint

        # Optimizer setup
        print("Optimizer - ")
        model = Nonconvex.Model(obj)
        Nonconvex.addvar!(model, zeros(nels), ones(nels), init = x0)
        Nonconvex.add_ineq_constraint!(model, constr)
        optimizer = Nonconvex.optimize(model, NLoptAlg(:LD_MMA), x0; options=NLoptOptions())
        println("Done")

        # write topology to file
        FEAparameters.fileID["topologies"][:, :, i] = quad(FEAparameters.meshSize..., optimizer.minimizer)

    end
    end

    return tempo

end

tempo = loopTO(FEAparams)

# time stats
# println(round.(tempo'))
totalTime  = sum(tempo)
println("total time: $(round(totalTime)) s; $(round(totalTime/3600;digits=1)) h; $(round(totalTime/86400;digits=1)) day(s)")
println("mean time: $(round(mean(tempo);digits=1)) s")
println("std of time: $(round(std(tempo);digits=1)) s")

# close file
close(FEAparams.fileID)

# save plots of samples
# @time plotSample2(FEAparams.quants, folderName, FEAparams)
@time plotSampleTest2(FEAparams.quants, folderName, FEAparams)

println("folder name: $folderName")

#= 
    # id = h5open("C:\\Users\\LucasKaoid\\Desktop\\datasets\\quickTOdata2-700", "r")
    # topo = read(id["topologies"])
    # inputs = read(id["inputs"])
    # forces = inputs["forces"]
    # bc = inputs["dispBoundConds"]
    # vf = inputs["VF"]

    # conds = read(id["conditions"])
    # disp = conds["disp"]
    # en = conds["energy"]
    # prin = conds["principalStress"]
    # stress = conds["stress_xy"]
    # vm = conds["vonMises"]
    # close(id)

    # fig = Figure(resolution=(1400,700))
    # heatmap(1:140,50:-1:1,vm[:,:,97]')
    # findmax([:,:,97])

    # BCs = [bc[:,:,i] for i in 1:700]
    # uniBC = unique(BCs)
    # length(uniBC)
    # filter(x[1,3]->x, BCs)

=#

# bizarre: 3 
# normal: 4 5 8 9 10 12 13 15 17 19 20 21 22 23 28 29 30