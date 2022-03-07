using Makie, TopOpt, Parameters, StatProfilerHTML, Printf, HDF5, Statistics
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent;
import GLMakie, Nonconvex
Nonconvex.@load NLopt;
include(".\\functionsQuickTO.jl")
println("loaded modules")


# struct with general parameters
@with_kw mutable struct FEAparameters
    quants::Int = 20 # number of problems
    problems::Any = Array{Any}(undef, quants) # store FEA problem structs
    V::Array{Real} = [0.3+rand()*0.6 for i in 1:quants] # volume fraction
    meshSize::Tuple{Int, Int} = (140, 50) # Size of rectangular mesh
    elementIDarray::Array{Int} = [i for i in 1:prod(meshSize)] # Vector that lists element IDs
    # matrix with element IDs in their respective position in the mesh
    elementIDmatrix::Array{Int,2} = convert.(Int, quad(meshSize...,[i for i in 1:prod(meshSize)]))
    # hdf5 file to store data
    fileID = createFile(quants,meshSize...)
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
        dispBC = zeros(Int, FEAparameters.meshSize)'

        #
            # cellType = Type of element (CPS4 = linear quadrilateral)
            # nodeSets = dictionary mapping strings to vectors of integers. The vector groups 
                # node IDs that can be later referenced by the name in the string
            if rand() > 0.5
                # clamp a side
                nodeSets, dispBC = simpleSupps!("rand", dispBC, FEAparams)
            else
                # position pins randomly
                nodeSets, dispBC = randSupps!(nels, FEAparams, dispBC)
            end
            # Dictionary mapping strings to vectors of tuples of Int and Float. The
                # string contains a name. It refers to
                # a group of nodes defined in nodeSets. The tuples inform the displacement (Float) applied to a
                # a certain DOF (Int) of the nodes in that group. This is used to apply
                # Dirichlet boundary conditions.
            nodeDbcs = Dict("supps" => [(1, 0.0), (2, 0.0)])
            # lpos has the IDs of the loaded nodes
            # Fx and Fy are matrices of the size of the mesh.
                # Non-zero values represent load components on the
                # nodes of that element
            lpos, Fx, Fy, loads = loadPos(nels, dispBC, FEAparameters)
            # Dictionary mapping integers to vectors of floats. The vector
                # represents a force applied to the node with
                # the respective integer ID.
            cLoads = Dict(lpos[1] => loads[1,:])
            merge!(cLoads, Dict(lpos[2] => loads[1,:]))
            merge!(cLoads, Dict(lpos[3] => loads[1,:]))
            merge!(cLoads, Dict(lpos[4] => loads[1,:]))
            if length(lpos) > 4
                for pos in 5:length(lpos)
                    pos == 5 && (global ll = 2)
                    merge!(cLoads, Dict(lpos[pos] => loads[ll,:]))
                    pos % 4 == 0 && (global ll += 1)
                end
            end
            # faceSets = Similar to nodeSets, but groups faces of cells (elements)
            # dLoads = Dictionary mapping strings to floats. The string refers to a group of cell faces
                # (element faces (or sides?)) defined in faceSets. The float is the value of a traction
                # applied to the faces inside that group.

            # Create TopOpt problem from inpCont struct
            problem = InpStiffness(InpContent(
                nodeCoords, "CPS4", cells, nodeSets, cellSets, 210e3, 0.3,
                0.0, nodeDbcs, cLoads,
                Dict("uselessFaces" => [(1,1)]), Dict("uselessFaces" => 0.0)))

            FEAparameters.problems[i] = problem

        #


        # FEA
        print("FEA - ")
        solver = FEASolver(Direct, problem; xmin=1e-6, penalty=TopOpt.PowerPenalty(3.0))
        solver()
        disp = solver.u

        #### write data to file
        print("Writing initial data - ")
        # write volume fraction to file
        FEAparameters.fileID["inputs"]["VF"][:,:,i] = fill(FEAparameters.V[i], (FEAparameters.meshSize[2], FEAparameters.meshSize[1]))
        # write displacements boundary conditions to file
        FEAparameters.fileID["inputs"]["dispBoundConds"][:,:,i] = dispBC
        # write Fx to file
        FEAparameters.fileID["inputs"]["Fx"][:,:,i] = Fx
        # write Fy to file
        FEAparameters.fileID["inputs"]["Fy"][:,:,i] = Fy
        # write displacements to file
        writeDisp(FEAparameters.fileID, i, disp, FEAparameters)
        # write stresses to file and verify material linearity
        writeStresses(nels, FEAparameters, disp, i, 210e3, 0.3)

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
        optimizer = Nonconvex.optimize(model, alg, x0; options=options)
        println("Done")

        # write topology to file
        FEAparameters.fileID["topologies"][:, :, i] = quad(FEAparameters.meshSize..., optimizer.minimizer)

    end
    end

    return tempo

end

tempo = loopTO(FEAparams)
println(round.(tempo'; digits=3))
println("total time: $(round(sum(tempo);digits=3)) s")
println("mean time: $(round(mean(tempo);digits=3)) s")

close(FEAparams.fileID)

# save plots of samples
@time plotSample(FEAparams.quants)


##### paralelizacao
##### quantidade de cargas e suportes?