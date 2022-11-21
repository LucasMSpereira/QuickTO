# functions used to write data to HDF5 files

# write stresses, principal components and strain energy density to file
function writeConds(fileID, vm, σ, principals, strainEnergy, problemID, FEAparams)

  fileID["conditions"]["vonMises"][:,:,problemID] = vm
  fileID["conditions"]["stress_xy"][:, :, problemID] = quad(FEAparams.meshSize...,σ)
  fileID["conditions"]["principalStress"][:, :, 2*problemID-1] = quad(FEAparams.meshSize...,principals[:, 1])
  fileID["conditions"]["principalStress"][:, :, 2*problemID] = quad(FEAparams.meshSize...,principals[:, 2])
  fileID["conditions"]["energy"][:,:,problemID] = strainEnergy

end

# write displacements to file
function writeDisp(quickTOdata, problemID, disp, FEAparams, numCellNode)
  dispScalar = Array{Real}(undef, prod(FEAparams.meshSize))
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(2), Lagrange{2,RefCube,ceil(Int, numCellNode/7)}())
  global el = 1
  # loop in elements
  for cell in CellIterator(FEAparams.problems[problemID].ch.dh)
    reinit!(cellValue, cell)
    # interpolate displacement (u, v) of element center based on nodal displacements.
    # then take the norm of this center displacement to associate a scalar to each element
    dispScalar[el] = norm(function_value(cellValue, 1, disp[celldofs(cell)]))
    global el += 1
  end
  # reshape to represent mesh
  dispScalar = quad(FEAparams.meshSize..., dispScalar)
  # add to dataset
  quickTOdata["conditions"]["disp"][:,:,problemID] = dispScalar
end

# write displacements to file
function writeDispComps(quickTOdata, problemID, disp, FEAparams, numCellNode)
  dispInterp = Array{Real}(undef, prod(FEAparams.meshSize), 2)
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(2), Lagrange{2, RefCube, ceil(Int, numCellNode/7)}())
  el = 1
  for cell in CellIterator(FEAparams.problems[problemID].ch.dh) # loop in elements
    reinit!(cellValue, cell)
    # interpolate displacement (u, v) of element center based on nodal displacements.
    dispInterp[el, :] = function_value(cellValue, 1, disp[celldofs(cell)])
    el += 1
  end
  # add to dataset
  quickTOdata["disp"][:, :, 2 * problemID - 1] = quad(FEAparams.meshSize..., dispInterp[:, 1])
  quickTOdata["disp"][:, :, 2 * problemID] = quad(FEAparams.meshSize..., dispInterp[:, 2])
  return dispInterp
end

# write data to new HDF5. Use depends on context
function writeToHDF5(fileID, disp, vf, supp, force, topology)
  for i in axes(vf, 1)
    fileID["topologies"][:, :, i] = topology[:, :, i]
    fileID["inputs"]["VF"][i] = vf[i]
    fileID["inputs"]["dispBoundConds"][:, :, i] = supp[:, :, i]
    fileID["inputs"]["forces"][:, :, i] = force[:, :, i]
    fileID["disp"][:, :, 2 * i - 1 : 2 * i] = disp[:, :, 2 * i - 1 : 2 * i]
  end
  close(fileID)
end