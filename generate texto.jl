# toy grid to get mesh info before problem is actually built
if elType == "CPS4"
    grid = generate_grid(Quadrilateral, FEAparameters.meshSize)
elseif elType == "CPS8"
    grid = generate_grid(QuadraticQuadrilateral, FEAparameters.meshSize)
end
numCellNodes = length(grid.cells[1].nodes) # number of nodes per cell/element

nels = prod(FEAparameters.meshSize) # number of elements in the mesh

if elType == "CPS4"
    nodeCoords, cells = mshData(FEAparameters.meshSize)
elseif elType == "CPS8"
    nodeCoords, cells = mshDataQuadratic(FEAparameters.meshSize)
end

cellSets = Dict(
    "SolidMaterialSolid" => FEAparameters.elementIDarray,
    "Eall"               => FEAparameters.elementIDarray,
    "Evolumes"           => FEAparameters.elementIDarray
)

i = 1 # counter for successful sample
tries = 0 # counter for attempted samples so far

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
# Each line in "forces" contains:
# [forceLine forceCol forceXcomponent forceYcomponent]
lpos, forces = loadPos(nels, dispBC, FEAparameters, grid)
# cLoads: dictionary mapping integers to vectors of floats. The vector
# represents a force applied to the node with
# the respective integer ID.
cLoads = Dict(lpos[1] => forces[1, 3:4])
[merge!(cLoads, Dict(lpos[c] => forces[1,3:4])) for c in 2:numCellNodes];
if length(lpos) > numCellNodes + 1
    for pos in (numCellNodes + 1):length(lpos)
        pos == (numCellNodes + 1) && (global ll = 2)
        merge!(cLoads, Dict(lpos[pos] => forces[ll, 3:4]))
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
        nodeCoords, elType, cells, nodeSets, cellSets,
        FEAparameters.V[i]*210e3, 0.3,
        0.0, Dict("supps" => [(1, 0.0), (2, 0.0)]), cLoads,
        Dict("unusedFaces" => [(1,1)]),
        Dict("unusedFaces" => 0.0)
    )
)

## FEA
# Define FEA solver with problem defined above
# TO penalty and minimal pseudo-density values are chosen
solver = FEASolver(
    Direct, problem; xmin=1e-6,
    penalty=TopOpt.PowerPenalty(3.0)
)
# Solve FEA problem, obtaining displacements
solver()
disp = copy(solver.u) # displacements

### Definitions for optimizer
# compliance
comp = TopOpt.Compliance(problem, solver)
# filtering to avoid checkerboard
filter = DensityFilter(solver; rmin=3.0)
# objective function
obj = x -> comp(filter(x));
# starting densities (VF everywhere)
x0 = fill(FEAparameters.V[i], nels)
# volume fraction of structure
volfrac = TopOpt.Volume(problem, solver)
# volume fraction constraint
constr = x -> volfrac(filter(x)) - FEAparameters.V[i]

### Optimizer setup
# create model
model = Nonconvex.Model(obj)
# add optimization variable
Nonconvex.addvar!(model, zeros(nels), ones(nels), init = x0)
# add volume constraint
Nonconvex.add_ineq_constraint!(model, constr)
# find optimum
optimizer = Nonconvex.optimize(
    model, NLoptAlg(:LD_MMA), x0; options=NLoptOptions()
)