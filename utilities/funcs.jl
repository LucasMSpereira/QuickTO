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

# transform binary support representation
# to dense 3x3 format
function binaryToDenseSupport(binarySupport)::Array{Float32, 3}
  denseSupport = zeros(Int64, (3, 3, size(binarySupport, 3)))
  for sample_ in axes(binarySupport, 3) # iterate in samples
    if all(==(1.0), binarySupport[:, :, sample_][:, 1])
      denseSupport[:, :, sample_] .= 4 # left clamped
    elseif all(==(1.0), binarySupport[:, :, sample_][end, :])
      denseSupport[:, :, sample_] .= 5 # bottom clamped
    elseif all(==(1.0), binarySupport[:, :, sample_][:, end] .== 1.0)
      denseSupport[:, :, sample_] .= 6 # right clamped
    elseif all(==(1.0), binarySupport[:, :, sample_][1, :])
      denseSupport[:, :, sample_] .= 7 # top clamped
    else # 3 random pins
      for (index, pinPos) in findall(==(1.0), binarySupport[:, :, sample_]) |> enumerate
        denseSupport[index, :, sample_] .= pinPos[1], pinPos[2], 3
      end
    end
  end
  return denseSupport
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

function contextQuantity(goal::Symbol, files::Vector{String}, percent)::Int
  if goal == :train
    length(files) == 1 && return round(Int,datasetNonTestSize * 0.7 * percent)
    return abs(round(Int,
      datasetNonTestSize * 0.7 * percent - numSample(files[1 : max(1, end - 1)])
    ))
  elseif goal == :validate
    length(files) == 1 && return round(Int,datasetNonTestSize * 0.3 * percent)
    return abs(round(Int,
      datasetNonTestSize * 0.3 * percent - numSample(files[1 : max(1, end - 1)])
    ))
  elseif goal == :test
    return round(Int, 15504 * percent)
  end
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
# to access files for training/validation
function defineGroupFiles(metaData, goal)
  if goal != :test # if training or validating
    return DataLoader(1:length(metaData.files[goal]) |> collect; batchsize = 20)
  else
    return [datasetPath * "/data/test"]
  end
end

# add fourth dimension to Flux data (width, height, channels, batch)
dim4 = unsqueeze(; dims = 4)

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
    if size(mat, 1) < 4
      mat[2, 2] = 0
    else
      mat[2:(end - 1) , 2:(end - 1)] .= 0
    end
    # store maximum value of current ring (and its position relative to the center element)
    peaks[circle] = findmax(mat)
    center = round(Int, (side + 0.01) / 2)
    Δx[circle] = peaks[circle][2][2] - center
    Δy[circle] = center - peaks[circle][2][1]
  end
  maxVals = [peaks[f][1] for f in keys(peaks)]
  x̄ = Δx' * maxVals / sum(maxVals)
  ȳ = Δy' * maxVals / sum(maxVals)
  return x̄, ȳ, avgs

end

# convert loads from dataset to dense format
function forceMatrixToDense(sparseX, sparseY)::Array{Float32, 3}
  denseForce = zeros(Float32, (2, 4, size(sparseX, 3)))
  for sample_ in axes(sparseX, 3), (index, loadPos) in findall(!=(0.0), sparseX[:, :, sample_]) |> enumerate
    denseForce[index, :, sample_] .= loadPos[1], loadPos[2], sparseX[:, :, sample_][loadPos], sparseY[:, :, sample_][loadPos]
  end
  return denseForce
end

# build sparse force matrices from dense format
function forceToMat(force)
  forceXmatrix = zeros(FEAparams.meshMatrixSize)
  forceYmatrix = zeros(FEAparams.meshMatrixSize)
  forceXmatrix[force[1, 1] |> Int, force[1, 2] |> Int] = force[1, 3] # x component of first load
  forceXmatrix[force[2, 1] |> Int, force[2, 2] |> Int] = force[2, 3] # x component of second load
  forceYmatrix[force[1, 1] |> Int, force[1, 2] |> Int] = force[1, 4] # y component of first load
  forceYmatrix[force[2, 1] |> Int, force[2, 2] |> Int] = force[2, 4] # y component of second load
  return forceXmatrix, forceYmatrix
end

# print summary statistics of GAN parameters
function GANparamsStats(metaData)
  println("Generator:\n")
  vcat((Iterators.flatten.(getGen(metaData) |> cpu |> Flux.params) .|> collect)...) |> statsum
  println("Discriminator:\n")
  vcat((Iterators.flatten.(getDisc(metaData) |> cpu |> Flux.params) .|> collect)...) |> statsum
  return nothing
end

# Get pixel-wise correlations for each generator
# input channel in a certain split
function generatorCorrelation(
  gen::Chain, split::Symbol;
  additionalFiles = 0, VFcorrelation = false
)::Union{Vector{Matrix{Float32}}, Float32}
  # initialize arrays
  input = zeros(Float32, (51, 141, 3, 1))
  fakeTopology = zeros(Float32, (51, 141, 1, 1))
  iter = 0
  # batches of data split
  @inbounds for (genInput, _, _) in dataBatch(split, 200; extraFiles = additionalFiles)[1]
    iter += 1
    rand() < 0.1 && println(iter, "  ", timeNow())
    input = cat(input, genInput; dims = 4) # store batch input
    # store topology
    fakeTopology = cat(fakeTopology, genInput |> gpu |> gen |> cpu |> padGen; dims = 4)
  end
  # remove first samples (null initialization)
  input, fakeTopology = remFirstSample.((input, fakeTopology))
  if !VFcorrelation # if calculating VM and energy pixelwise correlations
    # VM-topology and energy-topology pixelwise correlations
    channelCorr = [zeros(Float32, (51, 141)) for _ in 1:2]
    @inbounds for ch in 2:3 # iterate in both channels
      @inbounds for i in axes(input, 1), j in axes(input, 2)
        # standardize data and calculate correlation for current "pixel"
        channelCorr[ch - 1][i, j] = Statistics.cor(
          StatsBase.standardize(StatsBase.ZScoreTransform, input[i, j, ch, :]),
          StatsBase.standardize(StatsBase.ZScoreTransform, fakeTopology[i, j, 1, :])
        )
      end
    end
    # discard last row and column because of difference
    # in size of generator input and output
    channelCorr = [corMat[1 : end - 1, 1 : end - 1] for corMat in channelCorr]
    # in test set, all samples are clamped at the top. these
    # points are constant in the output, causing NaN correlations
    if split == :test
      channelCorr = [corMat[2 : end, :] for corMat in channelCorr]
    end
    return channelCorr
  else # if calculating only VF correlations
    return Statistics.cor(
      StatsBase.standardize(StatsBase.ZScoreTransform, volFrac(fakeTopology)),
      StatsBase.standardize(StatsBase.ZScoreTransform, input[1, 1, 1, :])
    )
  end
end

# performance of trained generator is certain data split
function genPerformance(gen::Chain, dataSplit::Vector{String})
  if length(dataSplit) == 1
    sampleAmount = 15504
  else
    sampleAmount = numSample(dataSplit)
  end
  splitError = Dict(
    :topoSE => zeros(Float32, sampleAmount),
    :VFerror => zeros(Float32, sampleAmount),
    :compError => zeros(Float32, sampleAmount)
  )
  fakeVF, realVF, pastSample, globalID = 0f0, 0f0, 0, 0
  fakeComp, realComp = 0f0, 0f0
  for (fileIndex, filePath) in enumerate(dataSplit)
    println(fileIndex, "/", length(dataSplit), " ", timeNow())
    if length(dataSplit) == 1
      fileSize = sampleAmount
    else
      fileSize = numSample([dataSplit[fileIndex]])
    end
    # tensor initializers
    genInput = zeros(Float32, FEAparams.meshMatrixSize..., 3, 1); FEAinfo = similar(genInput)
    topology = zeros(Float32, FEAparams.meshMatrixSize..., 1, 1)
    # gather data from multiple files (or test file)
    denseDataDict, dataDict = denseInfoFromGANdataset(replace(filePath, "LucasK" => "k"), fileSize)
    # dataDict = readTopologyGANdataset(replace(filePath, "LucasK" => "k"))
    genInput, FEAinfo, topology = groupGANdata!(
      genInput, FEAinfo, topology, dataDict
    )
    # discard initializers
    genInput, FEAinfo, realTopology = remFirstSample.((genInput, FEAinfo, topology))
    for sample in 1:fileSize
      globalID = pastSample + sample
      fakeTopology = cpu(genInput[:, :, :, sample] |> dim4 |> gpu |> gen) # use generator
      splitError[:topoSE][globalID] = sum( # squared topology error
        (padGen(fakeTopology)[:, :, 1, 1] .- realTopology[:, :, 1, sample]) .^ 2
      )
      # relative volume fraction error
      fakeVF = volFrac(fakeTopology[:, :, 1, 1])[1]
      realVF = volFrac(realTopology[:, :, 1, sample])[1]
      splitError[:VFerror][globalID] = abs(fakeVF - realVF) / realVF
      # relative compliance error
      @suppress_err fakeComp = topologyCompliance(
        Float64(denseDataDict[:vf][sample]),
        Int.(denseDataDict[:denseSupport][:, :, sample]),
        Float64.(denseDataDict[:force][:, :, sample]),
        Float64.(fakeTopology[:, :, 1, 1])
      )
      realComp = dataDict[:compliance][sample]
      splitError[:compError][globalID] = abs(fakeComp - realComp) / realComp
      sample % 350 == 0 && @show sample
    end
    pastSample += fileSize
  end
  return splitError
end

# Get section and dataset IDs of sample
function getIDs(pathing)
  s = parse.(Int, split(pathing[findlast(x->x=='\\', pathing)+1:end]))
  return s[1], s[2]
end

# Identify non-binary topologies
function getNonBinaryTopos(forces, supps, vf, disp, top)
  bound = 0.35 # densities within (0.5 +/- bound) are considered intermediate
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

# initialize variables used to store loss histories
function initializeHistories(_metaData)
  if wasserstein # using wgan
    push!(_metaData.discDefinition.nnValues, :criticOutFake, 1, 0f0)
    push!(_metaData.discDefinition.nnValues, :criticOutReal, 1, 0f0)
    push!(_metaData.discDefinition.nnValues, :genDoutFake, 1, 0f0)
    push!(_metaData.discDefinition.nnValues, :mse, 1, 0f0)
  else
    push!(_metaData.discDefinition.nnValues, :discTrue, 1, 0f0)
    push!(_metaData.discDefinition.nnValues, :discFalse, 1, 0f0)
    push!(_metaData.genDefinition.nnValues, :foolDisc, 1, 0f0)
    push!(_metaData.genDefinition.nnValues, :mse, 1, 0f0)
    push!(_metaData.genDefinition.nnValues, :vfMAE, 1, 0f0)
  end
  return nothing
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

# log discriminator batch values
function logBatchDiscVals(metaData_, discTrueVal::Float32, discFalseVal::Float32)
  newSize = length(metaData_.discDefinition.nnValues[:discTrue]) + 1
  push!(metaData_.discDefinition.nnValues, :discTrue, newSize, discTrueVal)
  push!(metaData_.discDefinition.nnValues, :discFalse, newSize, discFalseVal)
end

# log generator batch values
function logBatchGenVals(metaData_, foolDisc, mse, vfMAE)
  newSize = length(metaData_.genDefinition.nnValues[:foolDisc]) + 1
  push!(metaData_.genDefinition.nnValues, :foolDisc, newSize, foolDisc)
  push!(metaData_.genDefinition.nnValues, :mse, newSize, mse)
  push!(metaData_.genDefinition.nnValues, :vfMAE, newSize, vfMAE)
end

# contextual logit binary cross-entropy
function logitBinCrossEnt(logits, label)
  return Flux.Losses.logitbinarycrossentropy(
    logits,
    fill(Float32(label), length(logits))
  )
end

# contextual logit binary cross-entropy.
# includes noisy label-smoothing
function logitBinCrossEntNoise(logits::Array{Float32, 2}, label::AbstractFloat)::Float32
  # @show mean(logits)
  if label < 0.5
    return Flux.Losses.logitbinarycrossentropy(
      logits,
      randBetween(0, 0.15; sizeOut = size(logits)) .|> Float32
    )
  else
    return Flux.Losses.logitbinarycrossentropy(
      logits,
      randBetween(0.85, 1.0; sizeOut = size(logits)) .|> Float32
    )
  end
end

# log losses from wgan model
function logWGANloss(
  metaData_, cOutFake::Float32, cOutReal::Float32,
  generatorDoutFake::Float32, MSEerror::Float32
)
  newSize = length(metaData_.discDefinition.nnValues[:criticOutFake]) + 1
  push!(metaData_.discDefinition.nnValues, :criticOutFake, newSize, cOutFake)
  push!(metaData_.discDefinition.nnValues, :criticOutReal, newSize, cOutReal)
  push!(metaData_.discDefinition.nnValues, :genDoutFake, newSize, generatorDoutFake)
  push!(metaData_.discDefinition.nnValues, :mse, newSize, MSEerror)
end

# compare problem throughput of OT and ML models
function methodThroughput(nSample::Int, topologyGANgen::Chain, quickTOgen::Chain)
  for sample in 1:nSample
    sample % round(Int, nSample/5) == 0 && @show sample
    # keep trying until sample isn't problematic
    while true
      # random problem
      problem, vf, force = randomFEAproblem(FEAparams)
      # build solver
      solver = FEASolver(Direct, problem; xmin = 1e-6, penalty = TopOpt.PowerPenalty(3.0))
      # obtain FEA solution
      solver()
      # determine von Mises field
      vm, energy, _, _ = calcCondsGAN(deepcopy(solver.u), 210e3 * vf, 0.33; dispShape = :vector)
      if checkSample(size(force, 1), vm, 3, force) # if sample isn't problematic
        ## standard TO
        comp = TopOpt.Compliance(solver) # compliance
        filter = DensityFilter(solver; rmin = 3.0) # filtering to avoid checkerboard
        obj = x -> comp(filter(PseudoDensities(x))) # objective
        x0 = fill(vf, FEAparams.nElements) # starting densities (VF everywhere)
        volfrac = TopOpt.Volume(solver)
        constr = x -> volfrac(filter(PseudoDensities(x))) - vf # volume fraction constraint
        model = Nonconvex.Model(obj) # create optimization model
        Nonconvex.addvar!( # add optimization variable
          model, zeros(FEAparams.nElements), ones(FEAparams.nElements), init = x0
        )
        Nonconvex.add_ineq_constraint!(model, constr) # add volume constraint
        # time standard optimization with MMA
        @timeit to "standard" Nonconvex.optimize(model, NLoptAlg(:LD_MMA), x0; options = NLoptOptions())
        ## ML models
        mlInput = solidify(fill(vf, FEAparams.meshMatrixSize), vm, energy) |> dim4 |> gpu
        # time topologyGAN generator
        @timeit to "U-SE-ResNet" topologyGANgen(mlInput)
        # time QuickTO generator
        @timeit to "QuickTO" quickTOgen(mlInput)
        break
      end
    end
  end
end

# estimate total number of lines in project so far
function numLines()
  sum(
    [
      filter(x -> occursin(".", x), readdir(projPath * "networks"; join = true)) .|> readlines .|> length
      filter(x -> occursin(".", x), readdir(projPath * "utilities"; join = true)) .|> readlines .|> length
      filter(x -> occursin(".", x), readdir(projPath * "utilities/IO"; join = true)) .|> readlines .|> length
      filter(x -> occursin(".", x), readdir(projPath * "utilities/ML utils"; join = true)) .|> readlines .|> length
      filter(x -> occursin(".", x), readdir(projPath * "docs/old"; join = true)) .|> readlines .|> length
      filter(x -> occursin(".", x), readdir(projPath * "docs/old/DataAnalysis"; join = true)) .|> readlines .|> length
      filter(x -> occursin(".", x), readdir(projPath * "docs/old/generateDataset"; join = true)) .|> readlines .|> length
      filter(x -> occursin(".", x), readdir(projPath * "docs/old/loadCNN"; join = true)[2:end]) .|> readlines .|> length
    ]
  )
end

# normalize values of array between -1 and 1
function normalizeVals(x)::Array{Float32}
  maxVal = maximum(x)
  minVal = minimum(x)
  if maxVal == minVal
    return x
  else
    return map(e -> 2 / (maxVal - minVal + 1e-10) * (e - maxVal) + 1, x)
  end
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

# pad batch of generator outputs
function padGen(genOut)
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

pen_l1(x::AbstractArray) = sum(abs, x)

pen_l2(x::AbstractArray) = sum(abs2, x)

# generate string representing optimizer
function printOptimizer(optimizer)
  optString = string(typeof(optimizer))
  return optString[findlast(".", optString)[1] + 1 : end]
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

# uniformly generate random Float64 between inputs
function randBetween(lower::Real, upper::Real; sizeOut = 1)::Array{Float64}
  output = zeros(sizeOut)
  return [lower + (upper - lower) * rand() for _ in keys(output)]
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
remFirstSample(x) = x[:, :, :, 2:end]

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

# check for variation of certain pseudo-densities
# across topologies in a fake batch
function sampleVariety(genBatchOut::Array{Float32, 4})::Float32
  means = mean(genBatchOut; dims = [1, 2, 3])
  return maximum(means) - minimum(means)
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

# slow down loss modulus increase
smoothSqrt(x::Float32)::Float32 = sign(x) * sqrt(abs(x))

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
  intSupp = round.(Int, supp)
  suppMatNode = zeros(FEAparams.meshMatrixSize)
  suppMatElement = zeros(FEAparams.meshSize |> reverse)
  if supp[1, end] == 4 # left clamped
    suppMatNode[:, 1] .= 1.0
    suppMatElement[:, 1] .= 1.0
  elseif supp[1, end] == 5 # bottom clamped
    suppMatNode[end, :] .= 1.0
    suppMatElement[end, :] .= 1.0
  elseif supp[1, end] == 6 # right clamped
    suppMatNode[:, end] .= 1.0
    suppMatElement[:, end] .= 1.0
  elseif supp[1, end] == 7 # top clamped
    suppMatNode[1, :] .= 1.0
    suppMatElement[1, :] .= 1.0
  else # 3 random pins
    for line in axes(supp, 1)
      suppMatNode[intSupp[line, 1], intSupp[line, 2]] = 1.0
      suppMatElement[intSupp[line, 1], intSupp[line, 2]] = 1.0
    end
  end
  return suppMatNode, suppMatElement
end

# string with current time and date
timeNow() = replace(string(ceil(now(), Dates.Second)), ":" => "-")[6:end]

# characteristics of training for fixed epochs on
# certain percentage of the dataset
function trainStats(nEpochs, datasetPercentage, validFreq)
  trainEpochTime = 2.6 * nEpochs # hours spent training
  validationEpochTime = 1.3 * floor(nEpochs/validFreq) # hours spent validating
  testTime = 2.6 * 15504 / (datasetNonTestSize * 0.7) # hours spent testing 
  # estimated time in hours
  tTime = round(
    (trainEpochTime + validationEpochTime + testTime) * datasetPercentage;
    digits = 1
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
  for dPercent in Iterators.flatten((0.01:0.01:0.1, 0.2:0.1:1.0))
    println(rpad("$(round(Int, dPercent * 100))%", 7),
      round <| (Int, (days * 24) / (dPercent * (2.6 + 1.3/validFreq)))..., " epochs"
    )
  end
end

# notation analogue to |>, but works with multiple arguments
<|(f, args...) = f(args...)