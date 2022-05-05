# modules
using Makie, TopOpt, Parameters, StatProfilerHTML, Printf, HDF5, Statistics
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent;
import GLMakie, Nonconvex
Nonconvex.@load NLopt;
# function definitions
include("./QTOutils.jl")

# main function definition
function loopTO(FEAparameters, elType, runID)
    
    # each section: create file, generate data, save file
    for sec in 1:FEAparameters.section
        
        # hdf5 file to store data
        fileID = createFile(FEAparameters.quants, sec, runID, FEAparameters.meshSize...)
        
        # toy grid to get mesh info before problem is actually built
        if elType == "CPS4"
            grid = generate_grid(Quadrilateral, FEAparameters.meshSize)
        elseif elType == "CPS8"
            grid = generate_grid(QuadraticQuadrilateral, FEAparameters.meshSize)
        end
        numCellNodes = length(grid.cells[1].nodes) # number of nodes per cell/element
        
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
        
        tempo = zeros(FEAparameters.quants+1) # record time taken per sample
        i = 1 # counter for successful sample
        tries = 0 # counter for attempted samples so far
        # loop in samples of current section (file)
        while i <= FEAparameters.quants
            tempo[i] = @elapsed begin
                tries += 1
                print("Section $sec/$(FEAparameters.section)           ")
                print("Sample $i/$(FEAparameters.quants)            ")
                println("Progress: $(round(Int, (i+(sec-1)*FEAparameters.section)/(FEAparameters.quants*FEAparameters.section)*100))%")
                # print("Attempts $tries             ")
                # print("Discard rate: $(round(Int, (1-i/tries)*100))%          ")
                # print("Total time: $(round(sum(tempo)/3600;digits=1)) h            ")
                # println("Average sample time: $(round(Int, sum(tempo)/i)) s")
                
                # integer matrix representing displacement boundary conditions (supports):
                # 0: free element
                # 1: element restricted in the x direction ("roller")
                # 2: element restricted in the y direction ("roller")
                # 3: element restricted in both directions ("pinned"/"clamped")
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
                lpos, forces = loadPos(nels, dispBC, FEAparameters, grid)
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
                
                # faceSets = Similar to nodeSets, but groups faces of cells (elements)
                
                # dLoads = Dictionary mapping strings to floats. The string refers to a group of cell faces
                # (element faces/sides) defined in faceSets. The float is the value of a traction
                # applied to the faces inside that group.
                
                # nodeDbcs = Dictionary mapping strings to vectors of tuples of Int and Float. The
                # string contains a name. It refers to a group of nodes defined in nodeSets.
                # The tuples inform the displacement (Float) applied to a
                # a certain DOF (Int) of the nodes in that group. This is used to apply
                # Dirichlet boundary conditions.
                
                # cellType = Type of element (CPS4 = linear quadrilateral)
                
                # Create TopOpt problem from inpCont struct with randomized inputs defined above
                problem = InpStiffness(
                    InpContent(
                        nodeCoords, elType, cells, nodeSets, cellSets,  FEAparameters.V[i]*210e3, 0.3,
                        0.0, Dict("supps" => [(1, 0.0), (2, 0.0)]), cLoads,
                        Dict("uselessFaces" => [(1,1)]), Dict("uselessFaces" => 0.0)
                    )
                )

                FEAparameters.problems[i] = problem
                
                # FEA
                solver = FEASolver(Direct, problem; xmin=1e-6, penalty=TopOpt.PowerPenalty(3.0))
                solver()
                disp = copy(solver.u) # displacements
                
                # calculate conditional values (von Mises, σ_xy, σᵢ, strain energy density)
                # vm, σ, principals, strainEnergy = calcConds(nels, FEAparameters, disp, i, FEAparameters.V[i]*210e3, 0.3, numCellNodes)
                vm, _, _, _ = calcConds(nels, FEAparameters, disp, i, FEAparameters.V[i]*210e3, 0.3, numCellNodes)
                
                # check for problematic sample.
                # if that's the case, go to next iteration without saving current data to file
                sampleQuality = checkSample(size(forces,1), vm, i, 3, forces)
                if !sampleQuality
                    continue
                end
                
                #### write data to file
                # volume fraction
                fileID["inputs"]["VF"][i] = FEAparameters.V[i]
                # displacement boundary conditions
                fileID["inputs"]["dispBoundConds"][:,:,i] = dispBC
                # forces
                fileID["inputs"]["forces"][:,:,i] = forces
                # norm of interpolated displacement of element center
                # writeDisp(fileID, i, disp, FEAparameters, numCellNodes)
                writeDispComps(fileID, i, disp, FEAparameters, numCellNodes)
                # write stresses to file
                # writeConds(fileID, vm, σ, principals, strainEnergy, i, FEAparameters)
                
                # Definitions for optimizer
                comp = TopOpt.Compliance(problem, solver) # compliance
                filter = DensityFilter(solver; rmin=3.0) # filtering to avoid checkerboard
                obj = x -> comp(filter(x)); # objective
                x0 = fill(FEAparameters.V[i], nels) # starting densities (VF everywhere)
                volfrac = TopOpt.Volume(problem, solver)
                constr = x -> volfrac(filter(x)) - FEAparameters.V[i] # volume fraction constraint
                
                # Optimizer setup
                model = Nonconvex.Model(obj) # create model
                Nonconvex.addvar!(model, zeros(nels), ones(nels), init = x0) # add optimization variable
                Nonconvex.add_ineq_constraint!(model, constr) # add (volume) constraint
                optimizer = Nonconvex.optimize(model, NLoptAlg(:LD_MMA), x0; options=NLoptOptions()) # find optimum
                
                # write topology to file
                fileID["topologies"][:, :, i] = quad(FEAparameters.meshSize..., optimizer.minimizer)
                
                i += 1
                
            end
        end
        println()
        # close file
        close(fileID)
    end
    
end


# struct with general parameters
@with_kw mutable struct FEAparameters
    quants::Int = 1 # number of TO problems per section
    V::Array{Real} = [0.4+rand()*0.5 for i in 1:quants] # volume fractions
    problems::Any = Array{Any}(undef, quants) # store FEA problem structs
    meshSize::Tuple{Int, Int} = (140, 50) # Size of rectangular mesh
    elementIDarray::Array{Int} = [i for i in 1:prod(meshSize)] # Vector that lists element IDs
    # matrix with element IDs in their respective position in the mesh
    elementIDmatrix::Array{Int,2} = convert.(Int, quad(meshSize...,[i for i in 1:prod(meshSize)]))
    section::Int = 1 # Number of dataset HDF5 files with "quants" samples each
end
FEAparams = FEAparameters()

# Identify current run
# runID = rand(0:99999)

# generate "FEAparams.section" dataset HDF5 files, each containing "FEAparams.quants" random samples
# @time loopTO(FEAparams, "CPS4", runID)

# C:/Users/LucasKaoid/Desktop/datasets/

# 5 sections of 50 samples each: estimative of 20h for 1e4 samples (serialized, ~0.002h/sample = ~7.2s/sample). more than 50e3/week

# save plots of samples
# @time plotSamples(FEAparams)