# Functions for FEA post-processing

using Ferrite, Parameters, HDF5, LinearAlgebra, Glob

# calculate stresses, principal components and strain energy density
function calcConds(nels, FEAparams, disp, problemID, e, v, numCellNode)
  # "Programming the finite element method", 5. ed, Wiley, pg 35
  state = "stress"
  principals = Array{Real}(undef, nels, 2) # principal stresses
  global σ = Array{Real}(undef, nels) # stresses
  global strainEnergy = zeros(FEAparams.meshSize)' # strain energy density in each element
  global vm = zeros(FEAparams.meshSize)' # von Mises for each element
  global centerDispGrad = Array{Real}(undef, nels, 2)
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(2), Lagrange{2,RefCube,ceil(Int, numCellNode/7)}())
  global el = 1
  # determine stress-strain relationship dee according to 2D stress type
  if state == "strain"
    # plane strain
    dee = e*(1 - v)/((1 + v)*(1 - 2 * v))*
      [1 v/(1 - v) 0;
      v/(1 - v) 1 0;
      0 0 (1 - 2*v)/(2*(1 - v))]
  
  elseif state == "stress"
    # plane stress
    dee = e/(1-v^2)*[
    1 v 0
    v 1 0
    0 0 (1-v)/2
    ]
  elseif state == "axisymmetric"
    
    dee = e*(1 - v)/((1 + v)*(1 - 2 * v))*
    [1 v/(1 - v) 0 v/(1 - v);
    v/(1 - v) 1 0 v/(1 - v);
    0 0 (1 - 2*v)/(2*(1 - v)) 0;
    v/(1 - v) v/(1 - v) 0 1]
  else
    println("Invalid stress state.")
  end
  # loop in elements
  for cell in CellIterator(FEAparams.problems[problemID].ch.dh)
    reinit!(cellValue, cell)
    # interpolate gradient of displacements on the center of the element
    global centerDispGrad = function_symmetric_gradient(cellValue, 1, disp[celldofs(cell)])
    # use gradient components to build strain vector ([εₓ ε_y γ_xy])
    global ε = [
      centerDispGrad[1,1]
      centerDispGrad[2,2]
      centerDispGrad[1,2]+centerDispGrad[2,1]
    ]
    # use constitutive model to calculate stresses in the center of current element
    global stress = dee*ε
    # take norm of stress vector to associate a scalar to each element
    global σ[el] = norm(stress)
    # extract principal stresses
    global principals[el,:] .= sort(eigvals([stress[1] stress[3]; stress[3] stress[2]]))
    elPos = findfirst(x->x==el,FEAparams.elementIDmatrix)
    # build matrix with (center) von Mises value for each element
    global vm[elPos] = sqrt(stress'*[1 -0.5 0; -0.5 1 0; 0 0 3]*stress)
    global strainEnergy[elPos] = (1+v)*(stress[1]^2+stress[2]^2+2*stress[3]^2)/(2*e) - v*(stress[1]+stress[2])^2/(2*e)
    global el += 1
  end
  return vm, σ, principals, strainEnergy
end

# calculate stresses, principal components and strain energy density
function calcVM(nels, FEAparams, disp, e, v)
  # "Programming the finite element method", 5. ed, Wiley, pg 35
  state = "stress"
  vm = zeros(FEAparams.meshSize)' # von Mises for each element
  centerDispGrad = Array{Real}(undef, nels, 2)
  numCellNode = 4 # number of nodes per cell/element
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(2), Lagrange{2,RefCube,ceil(Int, numCellNode/7)}())
  el = 1
  # determine stress-strain relationship dee according to 2D stress type
  if state == "strain"
    # plane strain
    dee = e*(1 - v)/((1 + v)*(1 - 2 * v))*
      [1 v/(1 - v) 0;
      v/(1 - v) 1 0;
      0 0 (1 - 2*v)/(2*(1 - v))]
  
  elseif state == "stress"
    # plane stress
    dee = e/(1-v^2)*[
    1 v 0
    v 1 0
    0 0 (1-v)/2
    ]
  elseif state == "axisymmetric"
    
    dee = e*(1 - v)/((1 + v)*(1 - 2 * v))*
    [1 v/(1 - v) 0 v/(1 - v);
    v/(1 - v) 1 0 v/(1 - v);
    0 0 (1 - 2*v)/(2*(1 - v)) 0;
    v/(1 - v) v/(1 - v) 0 1]
  else
    println("Invalid stress state.")
  end
  # rearrange disp into vector
  dispX = reshape(disp[end:-1:1,:,1]',(1,:))
  dispY = reshape(disp[end:-1:1,:,2]',(1,:))
  disps = zeros(length(dispX)*2)
  for i in 1:length(dispX)
    disps[2*i-1] = dispX[i]
    disps[2*i] = dispY[i]
  end
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

function plotTopoIntermidiate(forces, supps, vf, disp, top)
  
end