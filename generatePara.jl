using Makie, TopOpt, Parameters, StatProfilerHTML, Printf, HDF5, Statistics
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent;
import GLMakie, Nonconvex
Nonconvex.@load NLopt;
include("./functionsQuickTO.jl")
include("./testFunctions.jl")

function loopTO(FEAparameters, elType, runID)
    
    fileIDs = Array{Any}(undef, Threads.nthreads())
    counter = ones(Int, Threads.nthreads())
    tries = zeros(Int, Threads.nthreads())
    tempo = zeros(FEAparameters.quants+1,Threads.nthreads())
    nodeSets = similar(fileIDs)
    forces = zeros(Int, (2,4,Threads.nthreads()))
    lpos = similar(fileIDs)
    cLoads = similar(fileIDs)
    solver = similar(fileIDs)
    disp = similar(fileIDs)
    vm = similar(fileIDs)
    σ = similar(fileIDs)
    principals = similar(fileIDs)
    strainEnergy = similar(fileIDs)
    = similar(fileIDs)
    = similar(fileIDs)
    = similar(fileIDs)
    = similar(fileIDs)
    = similar(fileIDs)
    = similar(fileIDs)

    Threads.@threads for sec in 1:FEAparameters.section
        
        # hdf5 file to store data
        fileIDs[Threads.threadid()] = createFile(FEAparameters.quants, sec, runID, FEAparameters.meshSize...)
        
        # toy grid to get mesh info before problem is actually built
        if elType == "CPS4"
            grid = generate_grid(Quadrilateral, FEAparameters.meshSize)
        elseif elType == "CPS8"
            grid = generate_grid(QuadraticQuadrilateral, FEAparameters.meshSize)
        end
        numCellNodes = length(grid.cells[1].nodes)
        
        nels = prod(FEAparameters.meshSize) # number of elements in the mesh
        
        # nodeCoords = Vector of tuples with node coordinates
        # cells = Vector of tuples of integers. Each line refers to an element
        # and lists the IDs of its nodes
        if elType == "CPS4"
            nodeCoords, cells = mshData(FEAparameters.meshSize)
        elseif elType == "CPS8"
            nodeCoords, cells = mshDataQuadratic(FEAparameters.meshSize)
        end
        
        # Similar to nodeSets, but refers to groups of cells (FEA elements) 
        cellSets = Dict(
            "SolidMaterialSolid" => FEAparameters.elementIDarray,
            "Eall"               => FEAparameters.elementIDarray,
            "Evolumes"           => FEAparameters.elementIDarray
        )
        
        while counter[Threads.threadid()] <= FEAparameters.quants
            tempo[counter[Threads.threadid()], Threads.threadid()] = @elapsed begin
                tries[Threads.threadid()] += 1
                # print("Section $sec\t")
                # print("Sample $i\t")
                # print("Attempts $tries\t\t")
                # print("Discard rate: $(round(Int, (1-i/tries)*100))%\t\t")
                # print("Total time: $(round(sum(tempo)/3600;digits=1)) h\t\t")
                # println("Average sample time: $(round(Int, sum(tempo)/i)) s")
                println("1 $(Threads.threadid())")
                
                # integer matrix representing displacement boundary conditions (supports):
                # 0: free element
                # 1: element restricted in the x direction ("roller")
                # 2: element restricted in the y direction ("roller")
                # 3: element restricted in both directions ("pinned"/"clamped")
                dispBC = zeros(Int, (3,3,Threads.nthreads()))
                
                # nodeSets = dictionary mapping strings to vectors of integers. The vector groups 
                # node IDs that can be later referenced by the name in the string
                if rand() > 0.6
                    # "clamp" a side
                    nodeSets[Threads.threadid()], dispBC[:,:,Threads.threadid()] = simplePins!("rand", dispBC[:,:,Threads.threadid()], FEAparameters)
                else
                    # position pins randomly
                    nodeSets[Threads.threadid()], dispBC[:,:,Threads.threadid()] = randPins!(nels, FEAparameters, dispBC[:,:,Threads.threadid()], grid)
                end
                
                # lpos has the IDs of the loaded nodes.
                # each line in "forces" contains [forceLine forceCol forceXcomponent forceYcomponent]
                lpos[Threads.threadid()], forces[Threads.threadid()] = loadPos(nels, dispBC[:,:,Threads.threadid()], FEAparameters, grid)
                # Dictionary mapping integers to vectors of floats. The vector
                # represents a force applied to the node with
                # the respective integer ID.
                cLoads[Threads.threadid()] = Dict(lpos[Threads.threadid()][1] => forces[1,3:4, Threads.threadid()])
                [merge!(cLoads[Threads.threadid()], Dict(lpos[Threads.threadid()][c] => forces[1,3:4,Threads.threadid()])) for c in 2:numCellNodes];
                if length(lpos[Threads.threadid()]) > numCellNodes+1
                    for pos in (numCellNodes+1):length(lpos[Threads.threadid()])
                        pos == (numCellNodes+1) && (global ll = 2)
                        merge!(cLoads[Threads.threadid()], Dict(lpos[Threads.threadid()][pos] => forces[ll,3:4,Threads.threadid()]))
                        pos % numCellNodes == 0 && (global ll += 1)
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
                FEAparameters.problems[counter[Threads.threadid()], Threads.threadid()] = InpStiffness(
                    InpContent(
                        nodeCoords, elType, cells, nodeSets[Threads.threadid()], cellSets,  FEAparameters.V[counter[Threads.threadid()], Threads.threadid()]*210e3, 0.3,
                        0.0, Dict("supps" => [(1, 0.0), (2, 0.0)]), cLoads[Threads.threadid()],
                        Dict("uselessFaces" => [(1,1)]), Dict("uselessFaces" => 0.0)
                    )
                )

                # FEA
                solver[Threads.threadid()] = FEASolver(
                    Direct, FEAparameters.problems[counter[Threads.threadid()], Threads.threadid()]; xmin=1e-6, penalty=TopOpt.PowerPenalty(3.0)
                )
                solver[Threads.threadid()]()
                disp[Threads.threadid()] = copy(solver[Threads.threadid()].u)
                
                # calculate conditional values (scalars and princpal stress components)
                vm, σ, principals, strainEnergy = calcConds(nels, FEAparameters, disp, i, FEAparameters.V[i]*210e3, 0.3, numCellNodes)
                
                # check for problematic sample.
                # if that's the case, discard it and go to next iteration
                sampleQuality = checkSample(size(forces,1), vm, i, 3, forces)
                if !sampleQuality
                    continue
                end
                println("2 $(Threads.threadid())")
                
                #### write data to file
                # write volume fraction to file
                fileID["inputs"]["VF"][i] = FEAparameters.V[i]
                # write displacements boundary conditions to file
                fileID["inputs"]["dispBoundConds"][:,:,i] = dispBC
                # write forces to file
                fileID["inputs"]["forces"][:,:,i] = forces
                # write displacements to file
                writeDisp(fileID, i, disp, FEAparameters, numCellNodes)
                # write stresses to file and verify material linearity
                writeConds(fileID, vm, σ, principals, strainEnergy, i, FEAparameters)
                
                # Definitions for optimizer
                comp = TopOpt.Compliance(problem, solver) # compliance
                filter = DensityFilter(solver; rmin=3.0) # filtering to avoid checkerboard
                obj = x -> comp(filter(x)); # objective
                x0 = fill(FEAparameters.V[i], nels) # starting densities (VF everywhere)
                volfrac = TopOpt.Volume(problem, solver)
                constr = x -> volfrac(filter(x)) - FEAparameters.V[i] # volume fraction constraint
                
                # Optimizer setup
                model = Nonconvex.Model(obj)
                Nonconvex.addvar!(model, zeros(nels), ones(nels), init = x0)
                Nonconvex.add_ineq_constraint!(model, constr)
                optimizer = Nonconvex.optimize(model, NLoptAlg(:LD_MMA), x0; options=NLoptOptions())
                
                # write topology to file
                fileID["topologies"][:, :, i] = quad(FEAparameters.meshSize..., optimizer.minimizer)
                
                i += 1
                println("3 $(Threads.threadid())")
                
            end
        end
        println("4 $(Threads.threadid())")

        println()
        # close file
        close(fileID)
    end
    
end


# struct with general parameters
@with_kw mutable struct FEAparameters
    quants::Int = 1000 # number of TO problems
    V::Array{Real} = [0.3+rand()*0.6 for i in 1:quants, j in Threads.nthreads()] # volume fraction
    problems::Any = Array{Any}(undef, (quants, Threads.nthreads())) # store FEA problem structs
    meshSize::Tuple{Int, Int} = (140, 50) # Size of rectangular mesh
    elementIDarray::Array{Int} = [i for i in 1:prod(meshSize)] # Vector that lists element IDs
    # matrix with element IDs in their respective position in the mesh
    elementIDmatrix::Array{Int,2} = convert.(Int, quad(meshSize...,[i for i in 1:prod(meshSize)]))
    section::Int = 11 # Number of dataset HDF5 files with "quants" samples each
end
FEAparams = FEAparameters()

# Identify current run
runID = rand(0:9999)
mkpath("C:/Users/LucasKaoid/Desktop/datasets/data/fotos/$runID")

# generate "FEAparams.section" dataset HDF5 files, each containing "FEAparams.quants" random samples
@time loopTO(FEAparams, "CPS4", runID)
# 5 sections of 50 samples each: estimative of 20h for 1e4 samples (serialized, ~0.002h/sample = ~7.2s/sample). more than 50e3/week

# save plots of samples
# @time plotSample(FEAparams, runID)
# @time plotSampleTest(FEAparams.quants, folderName, FEAparams)

# [println("$(i-1)\t$(round(tempo[i];digits=3))") for i in keys(tempo)]