# functions used to write data to HDF5 files

function appendMetaDataHistories(; oldMetaData, recentMetaData)
  previousGenHist, previousDiscHist, _, validFreq = getValuesFromTxt(oldMetaData)
  open(datasetPath * "data/checkpoints/" * recentMetaData, "w") do id # open file
    content = readlines(id) # read each line
    # indices of lines to be used as reference
    previousHistory = findfirst(==("** Training:"), content)
  end
end

# generate PDF report about GANs
function GANreport(metaData)
  modelName = string(metaData.trainConfig.epochs) * "-" *
  string(round(percentageDataset * 100; digits = 1)) * "%-" *
  string(metaData.trainConfig.validFreq) * "-" *
  timeNow()
  # create directory to store all PDFs
  if runningInColab == false # if running locally
    path = projPath * "networks/GANplots/" * modelName
  else # if running in colab
    path = "./gdrive/MyDrive/dataset files/GAN saves/" * modelName
  end
  mkpath(path)
  # create pdf with line plots of validation loss histories
  plotGANValHist(
    metaData.lossesVals,
    metaData.trainConfig.validFreq,
    path, modelName
  )
  GANtestPlotsReport(modelName, metaData, path)
  writeGANmetaData(metaData; finalTxtPath = path)
  # combinePDFs(path, modelName * " report")
  return nothing
end

# save both GAN NNs to BSON files
function saveGANs(metaData, currentEpoch; finalSave = false)
  # transfer models to cpu
  cpuGenerator = cpu(metaData.generator)
  cpuDiscriminator = cpu(metaData.discriminator)
  # save models
  if runningInColab == false # if running locally
    BSON.@save datasetPath * "data/checkpoints/" * timeNow() * "-$(currentEpoch)gen.bson" cpuGenerator
    BSON.@save datasetPath * "data/checkpoints/" * timeNow() * "-$(currentEpoch)disc.bson" cpuDiscriminator
  else # if running in google colab
    BSON.@save "./gdrive/MyDrive/dataset files/GAN saves" * timeNow() * "-$(currentEpoch)gen.bson" cpuGenerator
    BSON.@save "./gdrive/MyDrive/dataset files/GAN saves" * timeNow() * "-$(currentEpoch)disc.bson" cpuDiscriminator
  end
  if !finalSave
    # bring models back to gpu, if training will continue
    metaData.generator = gpu(cpuGenerator)
    metaData.discriminator = gpu(cpuDiscriminator)
  end
  return nothing
end

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
function writeGANmetaData(metaData; finalTxtPath = " ")
  if runningInColab == false # if running locally
    if finalTxtPath == " "
      savePath = datasetPath * "data/checkpoints"
    else
      savePath = finalTxtPath
    end
  else # if running in colab
    savePath = "./gdrive/MyDrive/dataset files/GAN saves"
  end
  # open(savePath * "/" * timeNow() * "metaData.txt", "w") do id
  open(
    join([savePath, timeNow(), "metaData.txt"], "/", ""),
    "w"
  ) do id
    valF = metaData.trainConfig.validFreq
    # number of validations
    numVals = length(metaData.lossesVals[:genValHistory])
    write(id, "START TIME: $startTime\n\n")
    write(id, "********* CONFIGURATION METADATA\n")
    write(id, "\nPERCENTAGE OF DATASET: " * string(round(percentageDataset * 100; digits = 1)) * "%")
    trainSize = round(Int, datasetNonTestSize * 0.7 * percentageDataset)
    validateSize = round(Int, datasetNonTestSize * 0.3 * percentageDataset)
    testSize = 0
    write(id, "\n\tTRAIN: " * string(trainSize))
    write(id, "\n\tVALIDATE: " * string(validateSize))
    if length(metaData.lossesVals[:genTest]) > 0
      testSize = round(Int, 15504 * percentageDataset)
      write(id, "\n\tTEST: " * string(testSize))
    end
    write(id, "\n\tTOTAL: " * string(trainSize + validateSize + testSize))
    write(id, "\n\nOPTIMISERS:" * "\n\tGENERATOR: " *
      printOptimizer(metaData.genOptInfo.opt) * " " *
      sciNotation(metaData.genOptInfo.opt.eta, 1) * "\n\tDISCRIMINATOR: " *
      printOptimizer(metaData.discOptInfo.opt) * " " *
      sciNotation(metaData.discOptInfo.opt.eta, 1) * "\n"
    )
    write(id, "\nTRAINING: ")
    if typeof(metaData.trainConfig) == epochTrainConfig
      write(id, "FIXED NUMBER OF EPOCHS - ", string(metaData.trainConfig.epochs))
    else
      write(id, "EARLYSTOPPING")
      write(id, "\n\tINTERVAL BETWEEN CHECKS (VALIDATION EPOCHS): ", string(metaData.trainConfig.earlyStopQuant))
      write(id, "\n\tMINIMAL PERCENTAGE DROP IN LOSS: ", string(metaData.trainConfig.earlyStopPercent), "%")
    end
    write(id, "\n\tVALIDATION FREQUENCY (EPOCHS): ", valF |> string)
    if metaData.trainConfig.decay == 0
      write(id, "\n\tNO LEARNING RATE DECAY.\n")
    else
      write(id, "\n\tDECAY RATE: ", metaData.trainConfig.decay |> string)
      write(id, "\n\tDECAY FREQUENCY (EPOCHS): ", metaData.trainConfig.schedule |> string * "\n")
    end
    # write history of validation losses
    write(id, "\n********* VALIDATION LOSS HISTORIES\n\n")
    write(id, "EPOCH   GENERATOR      DISCRIMINATOR\n")
    for (epoch, genLoss, discLoss) in zip(
      valF:valF:valF * numVals,
      metaData.lossesVals[:genValHistory], metaData.lossesVals[:discValHistory]
    )
      write(id, rpad(epoch, 8) * rpad(sciNotation(genLoss, 3), 15) * sciNotation(discLoss, 3) * "\n")
    end
    if length(metaData.lossesVals[:genTest]) > 0
      write(id, "\n********* TEST LOSSES\n\n")
      write(id, "GENERATOR: " * sciNotation(metaData.lossesVals[:genTest][1], 3) * "\n")
      write(id, "DISCRIMINATOR: " * sciNotation(metaData.lossesVals[:discTest][1], 3) * "\n")
    end
    write(id, "\n********* SEPARATION OF FILES\n")
    for (dataSplit, name) in zip([:train, :validate], ["\n** Training:", "\n** Validation:"])
      write(id, name * "\n\n")
      for file in metaData.files[dataSplit]
        write(id, file)
        if dataSplit != :validate || file != metaData.files[dataSplit][end]
          write(id, "\n")
        end
      end
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