# General functions

using Ferrite, Parameters, HDF5, LinearAlgebra, Glob

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
function checkSample(numForces, vals, sample, quants, forces)
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

# Get section and dataset IDs from sample
function getIDs(pathing)
  s = parse.(Int, split(pathing[findlast(x->x=='\\', pathing)+1:end]))
  return s[1], s[2]
end

# Returns total number of samples across files in list
numSample(files) = sum([parse(Int, split(files[g][findlast(x->x=='\\', files[g])+1:end])[3]) for g in keys(files)])

# reshape vectors with element quantity to reflect mesh shape
function quad(nelx,nely,vec)
  # nelx = number of elements along x axis (number of columns in matrix)
  # nely = number of elements along y axis (number of lines in matrix)
  # vec = vector of scalars, each one associated to an element.
    # this vector is already ordered according to element IDs
  quadd=zeros(nely,nelx)
  for i in 1:nely
    for j in 1:nelx
      quadd[nely-(i-1),j] = vec[(i-1)*nelx+1+(j-1)]
    end
  end
  return quadd
end

# generate vector with n random and different integer values between 1 and val
function randDiffInt(n, val)
  global randVec = zeros(Int, n)
  randVec[1] = rand(1:val)
  for ind in 2:n
    global randVec[ind] = rand(1:val)
    while in(randVec[ind], randVec[1:ind-1])
      global randVec[ind] = rand(1:val)
    end
  end
  return randVec
end

# auxiliary print function
showVal(x) = println(round.(x;digits=4))

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