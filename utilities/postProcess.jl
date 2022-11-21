# Functions for FEA post-processing

using Ferrite, Parameters, HDF5, LinearAlgebra, Glob

# calculate stresses, principal components and strain energy density
function calcConds(nels, FEAparams, disp, problemID, e, v, numCellNode)
  # "Programming the finite element method", 5. ed, Wiley, pg 35
  state = "stress"
  # principal stresses
  principals = permutedims(zeros(FEAparams.meshSize..., 2), (2, 1, 3))
  σ = Array{Real}(undef, nels) # stresses
  strainEnergy = zeros(FEAparams.meshSize)' # strain energy density in each element
  vm = zeros(FEAparams.meshSize)' # von Mises for each element
  centerDispGrad = Array{Real}(undef, nels, 2)
  cellValue = CellVectorValues(
    QuadratureRule{2, RefCube}(2), Lagrange{2,RefCube,ceil(Int, numCellNode/7)}()
  )
  el = 1
  # determine stress-strain relationship dee according to 2D stress type
  dee = deeMat(state, e, v)
  vecDisp = dispVec(disp) # rearrange disp into vector
  # loop in elements
  for cell in CellIterator(FEAparams.problems[problemID].ch.dh)
    reinit!(cellValue, cell)
    # interpolate gradient of displacements on the center of the element
    centerDispGrad = function_symmetric_gradient(cellValue, 1, vecDisp[celldofs(cell)])
    # use gradient components to build strain vector ([εₓ ε_y γ_xy])
    ε = [
      centerDispGrad[1,1]
      centerDispGrad[2,2]
      centerDispGrad[1,2]+centerDispGrad[2,1]
    ]
    # use constitutive model to calculate stresses in the center of current element
    stress = dee*ε
    # take norm of stress vector to associate a scalar to each element
    σ[el] = norm(stress)
    # element cartesian position in mesh
    elPos = findfirst(x->x==el,FEAparams.elementIDmatrix)
    # extract principal stresses
    principals[elPos, :] .= sort(eigvals([stress[1] stress[3]; stress[3] stress[2]]))
    # build matrix with (center) von Mises value for each element
    vm[elPos] = sqrt(stress'*[1 -0.5 0; -0.5 1 0; 0 0 3]*stress)
    strainEnergy[elPos] = (1+v)*(stress[1]^2+stress[2]^2+2*stress[3]^2)/(2*e) - v*(stress[1]+stress[2])^2/(2*e)
    el += 1
  end
  return vm, σ, principals, strainEnergy
end

# calculate von Mises stress
function calcVM(nels, FEAparams, disp, e, v)
  # "Programming the finite element method", 5. ed, Wiley, pg 35
  state = "stress"
  vm = zeros(FEAparams.meshSize)' # von Mises for each element
  centerDispGrad = Array{Real}(undef, nels, 2)
  numCellNode = 4 # number of nodes per cell/element
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(2), Lagrange{2,RefCube,ceil(Int, numCellNode/7)}())
  el = 1
  # determine stress-strain relationship dee according to 2D stress type
  dee = deeMat(state, e, v)
  disps = dispVec(disp) # rearrange disp into vector
  # loop in elements
  for cell in CellIterator(FEAparams.problems[1].ch.dh)
    reinit!(cellValue, cell)
    # interpolate gradient of displacements on the center of the element
    centerDispGrad = function_symmetric_gradient(cellValue, 1, disps[celldofs(cell)])
    # use gradient components to build strain vector ([εₓ ε_y γ_xy])
    ε = [
      centerDispGrad[1,1]
      centerDispGrad[2,2]
      centerDispGrad[1,2]+centerDispGrad[2,1]
    ]
    # use constitutive model to calculate stresses in the center of current element
    stress = dee*ε
    # locate element in mesh
    elPos = findfirst(x->x==el,FEAparams.elementIDmatrix)
    # build matrix with (center) von Mises value for each element
    vm[elPos] = sqrt(stress'*[1 -0.5 0; -0.5 1 0; 0 0 3]*stress)
    el += 1
  end
  return vm
end