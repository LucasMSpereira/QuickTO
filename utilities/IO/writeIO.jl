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

# save information in GANmetaData struct in txt file
function writeLosses(metaData)
  open(datasetPath * "data/checkpoints/" * timeNow() * " metaData.txt", "w") do id
    valF = metaData.trainConfig.validFreq
    # number of validations
    numVals = length(metaData.lossesVals[:genValHistory])
    write(id, "********* CONFIGURATION METADATA\n")
    write(id, "\nPERCENTAGE OF DATASET: " * round(Int, percentageDataset * 100) |> string * "%\n")
    write(id, "\nOPTIMISER: " *
      string(typeof(metaData.optInfo.opt)) * " " * sciNotation(metaData.optInfo.opt.eta, 1)
    )
    write(id, "\nTRAINING: ")
    if typeof(metaData.trainConfig) == epochTrainConfig
      write(id, "FIXED NUMBER OF EPOCHS - ", metaData.trainConfig.epochs |> string)
    else
      write(id, "EARLYSTOPPING")
      write(id, "\n\tINTERVAL BETWEEN CHECKS (VALIDATION EPOCHS): ", metaData.trainConfig.earlyStopQuant |> string)
      write(id, "\n\tMINIMAL PERCENTAGE DROP IN LOSS: ", metaData.trainConfig.earlyStopPercent |> string, "%")
    end
    write(id, "\n\tVALIDATION FREQUENCY (EPOCHS): ", valF |> string)
    write(id, "\n\tCHECKPOINT FREQUENCY (EPOCHS): ", metaData.trainConfig.checkPointFreq |> string)
    if metaData.trainConfig.decay == 0
      write(id, "\n\tNO LEARNING RATE DECAY.\n")
    else
      write(id, "\n\tDECAY RATE: ", metaData.trainConfig.decay |> string)
      write(id, "\n\tDECAY FREQUENCY (EPOCHS): ", metaData.trainConfig.schedule |> string * "\n")
    end
    # write history of validation losses
    write(id, "\n********* VALIDATION LOSS HISTORIES\n\n")
    write(id, "EPOCH   GENERATOR      DISCRIMINATOR\n")
    for (epoch, genLoss, discLoss) in zip(valF:valF:valF * numVals, metaData.lossesVals[:genValHistory], metaData.lossesVals[:discValHistory])
      write(id, rpad(epoch, 8) * rpad(sciNotation(genLoss, 3), 15) * sciNotation(discLoss, 3) * "\n")
    end
    if length(metaData.lossesVals[:genTest]) > 0
      write(id, "\n********* TEST LOSSES\n\n")
      write(id, "GENERATOR: " * sciNotation(metaData.lossesVals[:genTest][1], 3) * "\n")
      write(id, "DISCRIMINATOR: " * sciNotation(metaData.lossesVals[:discTest][1], 3) * "\n")
    end
  end
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