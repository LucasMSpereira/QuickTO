using Makie, TopOpt, Parameters, StatProfilerHTML, Printf, HDF5, Statistics
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent;
import GLMakie, Nonconvex
Nonconvex.@load NLopt;
include(".\\functionsQuickTO2.jl")

folderName = "680" # choose dataset
i = 4 # base problem from dataset
mkpath("C:/Users/LucasKaoid/Desktop/datasets/$(folderName)/fotos/sample $i")

@with_kw mutable struct FEAparametersTest
  quants::Int = 30 # number of problems
  problems::Any = Array{Any}(undef, quants) # store FEA problem structs
  # V::Array{Real} = [0.3+rand()*0.6 for i in 1:quants] # volume fraction
  meshSize::Tuple{Int, Int} = (20, 20) # Size of rectangular mesh
  elementIDarray::Array{Int} = [i for i in 1:prod(meshSize)] # Vector that lists element IDs
  # matrix with element IDs in their respective position in the mesh
  elementIDmatrix::Array{Int,2} = convert.(Int, quad(meshSize...,[i for i in 1:prod(meshSize)]))
  # hdf5 file to store data
  # fileID = createFile(quants,meshSize...)
end
FEAparams = FEAparametersTest()

id = h5open("C:\\Users\\LucasKaoid\\Desktop\\datasets\\$(folderName)\\$(folderName)data2", "r")
vf = read(id["inputs"]["VF"])
forces = read(id["inputs"]["forces"])
BCs = read(id["inputs"]["dispBoundConds"])
close(id)


nels = prod(FEAparams.meshSize) # number of elements in the mesh
nodeCoords, cells = mshData(FEAparams.meshSize)

cellSets = Dict(
  "SolidMaterialSolid" => FEAparams.elementIDarray,
  "Eall"               => FEAparams.elementIDarray,
  "Evolumes"           => FEAparams.elementIDarray
)

variation = 30
volumes = range(0.3,0.9, length=variation)

for j in 1:variation

if BCs[:,:,i][1,3] > 3
  
  
  if BCs[:,:,i][1,3] == 4
    # left wall
    # clamped nodes
    firstCol = [(n-1)*(FEAparams.meshSize[1]+1) + 1 for n in 1:(FEAparams.meshSize[2]+1)]
    secondCol = firstCol .+ 1
    nodeSet = Dict("supps" => vcat(firstCol, secondCol))
  elseif BCs[:,:,i][1,3] == 6
    # right wall
    # clamped nodes
    firstCol = [(FEAparams.meshSize[1]+1)*n for n in 1:(FEAparams.meshSize[2]+1)]
    secondCol = firstCol .- 1
    nodeSet = Dict("supps" => vcat(firstCol, secondCol))
  elseif BCs[:,:,i][1,3] == 5
    # bottom
    # clamped nodes
    nodeSet = Dict("supps" => [n for n in 1:(FEAparams.meshSize[1]+1)*2])
  elseif BCs[:,:,i][1,3] == 7
    # top
    # clamped nodes
    nodeSet = Dict("supps" => [n for n in ((FEAparams.meshSize[1]+1)*(FEAparams.meshSize[2]-1)+1):((FEAparams.meshSize[1]+1)*((FEAparams.meshSize[2]+1)))])
  end
  
  
else
  grid = generate_grid(Quadrilateral, FEAparams.meshSize)
  myCells = [grid.cells[g].nodes for g in [FEAparams.elementIDmatrix[BCs[:,:,i][f, 1], BCs[:,:,i][f, 2]] for f in 1:size(BCs,1)]]
  pos = vec(reshape([myCells[ele][eleNode] for eleNode in 1:4, ele in size(BCs,1)], (:,1)))
  nodeSet = Dict("supps" => pos)
end

loadElID = [FEAparams.elementIDmatrix[convert(Int, forces[:,:,i][f, 1]), convert(Int, forces[:,:,i][f, 2])] for f in 1:size(forces[:,:,i],1)]
grid = generate_grid(Quadrilateral, FEAparams.meshSize)
myCells = [grid.cells[g].nodes for g in loadElID]
lpos = reshape([myCells[ele][eleNode] for eleNode in 1:4, ele in keys(loadElID)], (:,1))

cLoads = Dict(lpos[1] => forces[:,:,i][1,3:4])
[merge!(cLoads, Dict(lpos[c] => forces[:,:,i][1,3:4])) for c in 2:4];
if length(lpos) > 4
  for pos in 5:length(lpos)
    pos == 5 && (global ll = 2)
    merge!(cLoads, Dict(lpos[pos] => forces[:,:,i][ll,3:4]))
    pos % 4 == 0 && (global ll += 1)
  end
end

# vol = vf[i]
vol = volumes[j]

problem = InpStiffness(InpContent(
  nodeCoords, "CPS4", cells, nodeSet, cellSets, vol*210e3, 0.3,
  0.0, Dict("supps" => [(1, 0.0), (2, 0.0)]), cLoads,
  Dict("uselessFaces" => [(1,1)]), Dict("uselessFaces" => 0.0))
)

FEAparams.problems[i] = problem

# FEA
solver = FEASolver(Direct, problem; xmin=1e-6, penalty=TopOpt.PowerPenalty(3.0))
solver()
disp = copy(solver.u)
# disp = solver.u

# Definitions for optimizer
comp = TopOpt.Compliance(problem, solver) # compliance
filter = DensityFilter(solver; rmin=3.0) # filtering to avoid checkerboard
obj = x -> comp(filter(x)) # objective
x0 = fill(vol, nels) # starting densities (VF everywhere)
volfrac = TopOpt.Volume(problem, solver)
constr = x -> volfrac(filter(x)) - vol # volume fraction constraint

model = Nonconvex.Model(obj)
Nonconvex.addvar!(model, zeros(nels), ones(nels), init = x0)
Nonconvex.add_ineq_constraint!(model, constr)
optimizer = Nonconvex.optimize(model, NLoptAlg(:LD_MMA), x0; options=NLoptOptions())

vm, stress, prin1, prin2, strainEnergy = writeStressesTest(nels, FEAparams, disp, i, vol*210e3, 0.3)

miniPlot(vm, folderName, i, j, vol)
println(
    "$j  ",
    "maximum: ", round(maximum(vm)),
    "   average: ", round(mean(vm)),
    "   std: ", round(std(vm)),
    "   VF: ", round(vol; digits=2)
  )
end

# testSamplePlot(folderName, FEAparams, i)
# fieldnames(typeof(FEAparams.problems[i].inp_content.cloads))
# FEAparams.problems[i].inp_content.cloads.keys
# FEAparams.problems[i].inp_content.cloads.vals