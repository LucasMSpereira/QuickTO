# General functions

# return "adjacency graph" of solid elements
function adjSolid(topo, elementIDmatrix)
  # index positions of solid elements in binary topology
  solids = findall(x -> x == 1, topo)
  numSolids = length(solids)
  # initialize connectivity matrix
  connect = zeros(Int, (4, numSolids))
  # build element conectivity matrix (right-top-left-bottom)
  for ele in 1:numSolids
    if solids[ele][2] + 1 <= size(topo, 2) && topo[solids[ele][1], solids[ele][2] + 1] == 1
      # right
      connect[1, ele] = elementIDmatrix[solids[ele][1], solids[ele][2] + 1]
    end
    if solids[ele][1] - 1 >= 1 && topo[solids[ele][1] - 1, solids[ele][2]] == 1
      # top
      connect[2, ele] = elementIDmatrix[solids[ele][1] - 1, solids[ele][2]]
    end
    if solids[ele][2] - 1 >= 1 && topo[solids[ele][1], solids[ele][2] - 1] == 1
      # left
      connect[3, ele] = elementIDmatrix[solids[ele][1], solids[ele][2] - 1]
    end
    if solids[ele][1] + 1 <= size(topo, 1) && topo[solids[ele][1] + 1, solids[ele][2]] == 1
      # bottom
      connect[4, ele] = elementIDmatrix[solids[ele][1] + 1, solids[ele][2]]
    end
  end
  ### graph
  # initialize graph with vertices referring to solid elements
  g = MetaGraph(SimpleGraph(numSolids, 0))
  # include respective element ID in each node
  [set_prop!(g, node, :id, string(elementIDmatrix[solids[node]])) for node in 1:numSolids]
  set_indexing_prop!(g, :id) # use this property as index
  # add edges to the graph, according to neighborhood relationships between solid elements
  # loop in solid elements
  for ele in 1:numSolids
    # loop in neighbors of current solid element
    for neighbor in 1:size(connect, 1)
      # check if there's a neighbor in current direction
      if connect[neighbor, ele] != 0
        add_edge!(
          g,
          g[string(elementIDmatrix[solids[ele]]), :id],
          g[string(connect[neighbor, ele]), :id]
        )
      end
    end
  end
  return g
end

# get coordinates of centers of elements
function centerCoords(nels, problem)
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(2), Lagrange{2,RefCube,1}())
  centerPos = Array{Any}(undef, nels)
  global el = 1
  # loop in elements
  for cell in CellIterator(problem.ch.dh)
      reinit!(cellValue, cell)
      centerPos[el] = spatial_coordinate(cellValue, 1, getcoordinates(cell))
  global el += 1
  end
  xCoords = [centerPos[i][1] for i in 1:nels]
  yCoords = [centerPos[i][2] for i in 1:nels]
  return xCoords, yCoords
end

# check if sample was generated correctly (solved the FEA problem it was given and didn't swap loads)
function checkSample(numForces, vals, quants, forces)
  sProds = zeros(numForces)
  grads = zeros(2,numForces)
  avgs = similar(sProds)
  # get physical quantity gradients and average value around locations of loads
  for f in 1:numForces
    grads[1,f], grads[2,f], avgs[f] = estimateGrads(vals, quants, round.(Int,forces[f,1:2])...)
  end
  # calculate dot product between normalized gradient and respective normalized load to check for alignment between the two
  [sProds[f] = dot((grads[:,f]/norm(grads[:,f])),(forces[f,3:4]/norm(forces[f,3:4]))) for f in 1:numForces]
  # ratio of averages of neighborhood values of scalar field
  vmRatio = avgs[1]/avgs[2]
  # ratio of load norms
  loadRatio = norm(forces[1,3:4])/norm(forces[2,3:4])
  # ratio of the two ratios above
  ratioRatio = vmRatio/loadRatio
  magnitude = false
  # test alignment
  alignment = sum(abs.(sProds)) > 1.8
  if alignment
    # test scalar neighborhood averages against force norms
    magnitude = (ratioRatio < 1.5) && (ratioRatio > 0.55)
  end
  return alignment*magnitude
end

# among solid elements, get elements closest to each corner
function cornerPos(topo)
  # list with positions of elements closest to each corner (same order as loop below)
  closeEles = Array{CartesianIndex{2}}(undef, 4)
  # loop in corners
  for corner in ["bottomLeft" "bottomRight" "topRight" "topLeft"]
    # matrix to store distance of each active element to current corner
    dists = fill(1e12, size(topo))
    # get position of current corner
    corner == "bottomLeft" && (cornerPos = (size(topo, 1), 1))
    corner == "bottomRight" && (cornerPos = size(topo))
    corner == "topRight" && (cornerPos = (1, size(topo, 2)))
    corner == "topLeft" && (cornerPos = (1, 1))
    # loop in topology matrix
    for ele in keys(topo)
      # in distance matrix, position of active element is filled
      # with the respective (index) distance to current corner
      if topo[ele] == 1
        dists[ele] = sqrt(
          (ele[2] - cornerPos[2]) ^ 2 + (ele[1] - cornerPos[1]) ^ 2
        )
      end
    end
    # include position of closest element to current corner in list
    corner == "bottomLeft" && (closeEles[1] = findmin(dists)[2])
    corner == "bottomRight" && (closeEles[2] = findmin(dists)[2])
    corner == "topRight" && (closeEles[3] = findmin(dists)[2])
    corner == "topLeft" && (closeEles[4] = findmin(dists)[2])
  end
  return closeEles
end

# define 'groupFiles' variable in GANepoch!(). Used to get indices
# to acces files for training/validation
function defineGroupFiles(metaData, goal)
  if goal != :test # if training or validating
    return groupFiles = DataLoader(1:length(metaData.files[goal]) |> collect; batchsize = 13)
  else
    # if testing, variable 'groupFiles' has no effect.
    return groupFiles = [1] # Return placeholder
  end
end

# add fourth dimension to Flux data (width, height, channels, batch)
dim4 = unsqueeze(; dims = 5)

# discard first channel of 3D array
discardFirstChannel(x) = x[:, :, 2:end]

# identify samples with structural disconnection in final topology
function disconnections(topology, dataset, section, sample)
  # make sample topology binary
  topo = map(
    x -> x >= 0.5 ? 1 : 0,
    topology
  )
  # matrix with element IDs in mesh position
  elementIDmatrix = convert.(Int, quad(size(topo, 2), size(topo, 1), [i for i in 1:length(topo[:, :, 1])]))
  # solid elements closest to each corner (bottom left, then counter-clockwise)
  extremeEles = cornerPos(topo)
  # "adjacency graph" between solid elements
  g = adjSolid(topo, elementIDmatrix)
  # node IDs in graph referring to extreme elements
  nodeID = [g[string(elementIDmatrix[extremeEles[ele]]), :id] for ele in 1:length(extremeEles)]
  # possible pairs of extreme elements
  combsExtEles = collect(combinations(nodeID, 2))
  # A* paths between possible pairs of extreme elements
  paths = [a_star(g, combsExtEles[i]...) for i in 1:length(combsExtEles)]
  if prod(length.(paths)) == 0
    # Get list of elements visited in each path
    pathLists = pathEleList.(filter(x -> length(x) != 0, paths))
    topo[extremeEles] .+= 3 # indicate extreme elements in plot
    # IDs of nodes in paths connecting extreme elements (without repetition)
    if sum(length.(pathLists)) > 0
      nodePathsID = unique(cat(pathLists...; dims = 1))
      # List of element IDs in paths connecting extreme elements (without repetition)
      elementPathsID = [parse(Int, get_prop(g, nodePathsID[node], :id)) for node in keys(nodePathsID)]
      # Mesh positions of elements in paths connecting extreme elements (without repetition)
      elePathsPos = [findfirst(x -> x == elementPathsID[ele], elementIDmatrix) for ele in 1:length(elementPathsID)]
      topo[elePathsPos] .+= 2 # indicate elements in paths in plot
    end
    # create plot
    fig = Figure(;resolution = (1200, 400));
    heatmap(fig[1, 1], 1:size(topo, 1), size(topo, 2):-1:1, topo');
    # save image file
    save(datasetPath*"analyses/disconnection/problems imgs/$dataset $section $(string(sample)).png", fig)
  end
  #=
    product of lengths of A* paths connecting extreme elements
    in binary version of final topology. If null, this sample
    suffers from structural disconnection and a heatmap plot
    will be generated
  =#
  return prod(length.(paths))
end

# rearrange disp into vector
function dispVec(disp)
  dispX = reshape(disp[end : -1 : 1 , :, 1]', (1, :))
  dispY = reshape(disp[end : -1 : 1, :, 2]', (1, :))
  disps = zeros(length(dispX) * 2)
  for i in axes(dispX, 2)
    disps[2 * i - 1] = dispX[i]
    disps[2 * i] = dispY[i]
  end
  return disps
end

# estimate scalar gradient around element in mesh
function estimateGrads(vals, quants, iCenter, jCenter)
  peaks = Array{Any}(undef,quants)
  Δx = zeros(quants)
  Δy = zeros(quants)
  avgs = 0.0
  # pad original matrix with zeros along its boundaries to avoid index problems with kernel
  cols = size(vals,2)
  lines = size(vals,1)
  vals = vcat(vals, zeros(quants,cols))
  vals = vcat(zeros(quants,cols), vals)
  vals = hcat(zeros(lines+2*quants,quants), vals)
  vals = hcat(vals,zeros(lines+2*quants,quants))
  for circle in 1:quants
    # size of internal matrix
    side = 2*(circle+1) - 1
    # variation in indices
    delta = convert(Int,(side-1)/2)
    # build internal matrix
    mat = vals[(iCenter-delta+quants):(iCenter+delta+quants),(jCenter-delta+quants):(jCenter+delta+quants)]
    # calculate average neighborhood values
    circle == quants && (avgs = mean(filter(!iszero,mat)))
    # nullify previous internal matrix/center element
    if size(mat,1) < 4
      mat[2,2] = 0
    else
      mat[2:(end-1),2:(end-1)] .= 0
    end
    # store maximum value of current ring (and its position relative to the center element)
    peaks[circle] = findmax(mat)
    center = round(Int, (side+0.01)/2)
    Δx[circle] = peaks[circle][2][2] - center
    Δy[circle] = center - peaks[circle][2][1]
  end
  maxVals = [peaks[f][1] for f in keys(peaks)]
  x̄ = Δx'*maxVals/sum(maxVals)
  ȳ = Δy'*maxVals/sum(maxVals)
  return x̄, ȳ, avgs

end

# print summary statistics of GAN parameters
function GANparamsStats(metaData)
  println("Generator:\n")
  vcat((Iterators.flatten.(metaData.generator |> cpu |> Flux.params) .|> collect)...) |> statsum
  println("Discriminator:\n")
  vcat((Iterators.flatten.(metaData.discriminator |> cpu |> Flux.params) .|> collect)...) |> statsum
  return nothing
end

# Identify non-binary topologies
function getNonBinaryTopos(forces, supps, vf, disp, top)
  bound = 0.35 # densities within 0.5 +/- bound are considered intermediate
  boundPercent = 3 # allowed percentage of elements with intermediate densities
  intermQuant = length(filter(
      x -> (x > 0.5 - bound) && (x < 0.5 + bound),
      top
  ))
  intermPercent = intermQuant/length(top)*100
  # return intermPercent
  if intermPercent > boundPercent
      return intermPercent
  else
      return 0.0
  end
end

# get lists of hdf5 files to be used in training and validation
function getNonTestFileLists(trainValidateFolder, trainPercentage)
  # get list of files, and shuffle it
  filePaths = readdir(trainValidateFolder; join = true) |> shuffle!
  # amount of samples used in training
  datasetTrainSize = trainPercentage * datasetNonTestSize |> round
  trainingFiles = 0
  while true # loop including more files each time
    trainingFiles += 1
    # if size of training split was reached, break
    numSample(filePaths[1 : trainingFiles]) >= datasetTrainSize && break
  end
  return Dict( # create lists of files to be used in each split
    :train => filePaths[1:trainingFiles],
    :validate => filePaths[trainingFiles + 1 : end]
  )
end

# Get section and dataset IDs of sample
function getIDs(pathing)
  s = parse.(Int, split(pathing[findlast(x->x=='\\', pathing)+1:end]))
  return s[1], s[2]
end

function forceToMat(force)
  forceXmatrix = zeros(FEAparams.meshMatrixSize)
  forceYmatrix = zeros(FEAparams.meshMatrixSize)
  forceXmatrix[force[1, 1] |> Int, force[1, 2] |> Int] = force[1, 3] # x component of first load
  forceXmatrix[force[2, 1] |> Int, force[2, 2] |> Int] = force[2, 3] # x component of second load
  forceYmatrix[force[1, 1] |> Int, force[1, 2] |> Int] = force[1, 4] # y component of first load
  forceYmatrix[force[2, 1] |> Int, force[2, 2] |> Int] = force[2, 4] # y component of second load
  return forceXmatrix, forceYmatrix
end

# test if all "features" (forces and individual supports) aren't isolated (surrounded by void elements)
function isoFeats(force, supp, topo)
  pos = [
      Int(force[1, 1]) Int(force[1, 2])
      Int(force[2, 1]) Int(force[2, 2])
      supp[1, 1] supp[1, 2]
      supp[2, 1] supp[2, 2]
      supp[3, 1] supp[3, 2]
    ]
  # pad topology with zeros on all sides
  topoPad = vcat(topo, zeros(size(topo, 2))')
  topoPad = vcat(zeros(size(topoPad, 2))', topoPad)
  topoPad = hcat(zeros(size(topoPad, 1)), topoPad)
  topoPad = hcat(topoPad, zeros(size(topoPad, 1)))
  # total density in neighborhood of each feature
  neighborDens = zeros(size(pos, 1))
  # loop in features
  for feat in 1:size(pos, 1)
    # 3 elements in line above feature
    neighborDens[feat] += sum(topoPad[pos[feat, 1] - 1 + 1, pos[feat, 2] - 1  + 1:pos[feat, 2] + 1  + 1])
    # elements in each side of the feature
    neighborDens[feat] += topoPad[pos[feat, 1]  + 1, pos[feat, 2] - 1  + 1]
    neighborDens[feat] += topoPad[pos[feat, 1]  + 1, pos[feat, 2] + 1  + 1]
    # 3 elements in line below feature
    neighborDens[feat] += sum(topoPad[pos[feat, 1] + 1  + 1, pos[feat, 2] - 1  + 1:pos[feat, 2] + 1  + 1])
  end
  length(unique(pos[3:5, :])) == 1 && (neighborDens[3:5] .+= 1)
  # test density surrounding each feature against lower bound of 0.5/8 = 0.0625
  # i.e. surrounding density must accumulate to at least 0.5.
  # if at least one feature is isolated, function will return false
  return all(neighborDens .> 0.0625)
end

# contextual logit binary cross-entropy. "If" statement
# tries to avoid allocation of new vector of 1s or 0s
# when possible (far majority of times). It will then use instead
# constant global variables discBinaryReal and discBinaryFake
function logitBinCrossEnt(logits, label)
  if length(logits) != batchSize
    expected = fill(label |> Float64, size(logits))
  else
    expected = label == 1 ? discBinaryReal : discBinaryFake
  end
  return Flux.Losses.logitbinarycrossentropy(logits, expected)
end

# estimate total number of lines in project so far
function numLines()
  sum(
    [
      filter(x -> occursin(".", x), readdir(projPath * "/networks"; join = true)) .|> readlines .|> length
      filter(x -> occursin(".", x), readdir(projPath * "/utilities"; join = true)) .|> readlines .|> length
      filter(x -> occursin(".", x), readdir(projPath * "/utilities/IO"; join = true)) .|> readlines .|> length
      filter(x -> occursin(".", x), readdir(projPath * "/utilities/ML utils"; join = true)) .|> readlines .|> length
      filter(x -> occursin(".", x), readdir(projPath * "/docs/old"; join = true)) .|> readlines .|> length
      filter(x -> occursin(".", x), readdir(projPath * "/docs/old/DataAnalysis"; join = true)) .|> readlines .|> length
      filter(x -> occursin(".", x), readdir(projPath * "/docs/old/generateDataset"; join = true)) .|> readlines .|> length
      filter(x -> occursin(".", x), readdir(projPath * "/docs/old/loadCNN"; join = true)[2:end]) .|> readlines .|> length
    ]
  )
end

# normalize values of array between -1 and 1
function normalizeVals(x)
  maxVal = maximum(x)
  minVal = minimum(x)
  return map(e -> 2 / (maxVal - minVal) * (e - maxVal) + 1, x)
end

# Returns total number of samples across files in list
function numSample(files)
  if runningInColab == false # if running locally
    return sum([
      parse(Int, split(files[g][findlast(==('\\'), files[g]) + 1 : end])[3]) for g in keys(files)
    ])
  else # if running in colab
    return sum([
      parse(Int, split(files[g][findlast(==('/'), files[g]) + 1 : end])[3]) for g in keys(files)
    ])
  end
end

# pad output of generator
function padGen(genOut::Array{Float32, 4})::Array{Float32, 4}
  return cat(
    cat(genOut, zeros(Float32, (FEAparams.meshSize[2], 1, 1, size(genOut, 4))); dims = 2),
    zeros(Float32, (1, FEAparams.meshMatrixSize[2], 1, size(genOut, 4)));
    dims = 1
)
end

# get list of elements visited by a path
function pathEleList(aStar)
  list = zeros(Int, length(aStar) + 1)
  for ele in keys(list)
    ele == length(list) ? (list[ele] = aStar[ele - 1].dst) : (list[ele] = aStar[ele].src)
  end
  return list
end

# reshape vectors with element quantity to reflect mesh shape
function quad(nelx::Int, nely::Int, vec::Vector{<:Real})
  # nelx = number of elements along x axis (number of columns in matrix)
  # nely = number of elements along y axis (number of lines in matrix)
  # vec = vector of scalars, each one associated to an element.
    # this vector is already ordered according to element IDs
  quadd = zeros(nely, nelx)
  for i in 1:nely
    for j in 1:nelx
      quadd[nely - (i - 1), j] = vec[(i - 1) * nelx + 1 + (j - 1)]
    end
  end
  return quadd
end

# generate vector with n random and different integer values from 1 to val
function randDiffInt(n, val)
  randVec = zeros(Int, n)
  randVec[1] = rand(1:val)
  for ind in 2:n
    randVec[ind] = rand(1:val)
    while in(randVec[ind], randVec[1:ind-1])
      randVec[ind] = rand(1:val)
    end
  end
  return randVec
end

# remove first position along 4th dimension in 4D array
remFirstSample(x::Array{<:Real, 4})::Array{<:Real, 4} = x[:, :, :, 2:end]

# reshape output of discriminator
reshapeDiscOut(x) = dropdims(x |> transpose |> Array; dims = 2)

# Reshape output from stressCNN
function reshapeForces(predForces)
  forces = zeros(Float32, (2, 4))
  if length(predForces) != 4
    for col in 1:Int(length(predForces)/2)
      forces[:, col] .= predForces[2*col - 1 : 2*col]
    end
  else # if multi-output
    for col in eachindex(predForces)
      forces[:, col] .= predForces[col]
    end
  end
  return forces
end

# print real number in scientific notation
function sciNotation(num::Real, printDigits::Int)
  try
    num == 0 && return "0." * repeat("0", printDigits) * "E00"
    base10 = num |> abs |> log10 |> floor
    mantissa = round(abs(num) / 10 ^ base10; digits = printDigits)
    num < 0 && return "-$(mantissa)E$(Int(base10))"
    return "$(mantissa)E$(Int(base10))"
  catch
    @warn "sciNotation problem"
    return "0"
  end
end

showVal(x) = println(round.(x; digits = 4)) # auxiliary print function

# print order of layers and variation in data size along Flux chain
# problem with number print and recursion
function sizeLayers(myChain, input; currentLayer = 0)
  @show currentLayer
  if currentLayer == 0
    println("Input size: ", size(input))
    layer = 1
  else
    layer = copy(currentLayer)
  end
  for _ in 1:length(myChain) # iterate in NN layers
    if typeof(myChain[layer]) <: SkipConnection # recursion in case of skip connection
      sizeLayers(
        myChain[layer].layers,
        input |> myChain[1 : layer - 1];
        currentLayer = max(layer - 1, 1)
      )
    else # print size of output if not skip connection
      println(
        layer + currentLayer, ": ", typeof(myChain[layer]),
        "   -   output size: ", input |> myChain[1:layer] |> size
      )
      layer += 1
    end
  end
  return currentLayer
end

# concatenate multiple 2D or 3D arrays in the 3rd dimension
solidify(x...) = cat(x...; dims = 3)

# statistical summary of a numerical array
function statsum(arr)
  println(size(arr))
  data = reshape(arr, (1, :)) |> vec
  data |> summarystats |> print
  println("Standard deviation: ", sciNotation(std(data), 4))
  nonZeros = findall(x -> Float64(x) != 0.0, arr) |> length
  println(
    nonZeros,
    "/", length(arr),
    " (", round(nonZeros/length(arr)*100; digits = 1),
    "%) non-zero elements.\n"
  )
  return nothing
end
statsum(x...) = statsum.(x)

# use compact 3x3 support definition to create
# matrix used to train the GANs
function suppToBinary(supp)
  suppMat = zeros(FEAparams.meshMatrixSize)
  if supp[1, end] == 4 # left clamped
    suppMat[:, 1] .= 1.0
  elseif supp[1, end] == 5 # bottom clamped
    suppMat[end, :] .= 1.0
  elseif supp[1, end] == 6 # right clamped
    suppMat[:, end] .= 1.0
  elseif supp[1, end] == 7 # top clamped
    suppMat[1, :] .= 1.0
  else # 3 random pins
    for line in axes(supp, 1)
      suppMat[supp[line, 1], supp[line, 2]] = 1.0
    end
  end
  return suppMat
end

# string with current time and date
timeNow() = replace(string(ceil(now(), Dates.Second)), ":" => "-")

# characteristics of training for fixed epochs on
# certain percentage of the dataset
function trainStats(nEpochs, datasetPercentage, validFreq)
  tTime = round( # estimated time in hours
    (    # train               validate                     test
      5 * nEpochs + 2.5 * (floor(nEpochs/validFreq) + 15504 / (datasetNonTestSize * 0.3))
    ) * datasetPercentage
    ; digits = 1
  )
  println(tTime, " hour(s)   ", round(tTime / 24; digits = 1), " day(s)")
  # estimated distributions of samples in splits
  trainingAmount = round(Int, datasetNonTestSize * 0.7 * datasetPercentage)
  validationAmount = round(Int, datasetNonTestSize * 0.3 * datasetPercentage)
  testAmount = round(Int, 15504 * datasetPercentage)
  println("Amount of samples:")
  println("   Training: ", trainingAmount)
  println("   Validation: ", validationAmount)
  println("   Test: ", testAmount)
  println("   Total: ", trainingAmount + validationAmount + testAmount)
end
#= multiple dispatch of function above to suggest combinations of percentage
of dataset used and number of fixed epochs to train for certain amount
of time =#
function trainStats(validFreq, days)
  for dPercent in 0.1:0.1:1.0
    println(rpad("$(dPercent * 100 |> Int)%", 7),
      round <| (Int, (days * 24) / (dPercent * (5 + 2.5/validFreq)))..., " epochs"
    )
  end
end

# notation analogue to |>, but works with multiple arguments
<|(f, args...) = f(args...)