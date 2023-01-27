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
  # create pdf with line plots of validation loss histories
  modelName = GANfolderPath[end - 4 : end - 1]
  plotGANValHist(
    metaData.lossesVals,
    metaData.trainConfig.validFreq,
    modelName
  )
  GANtestPlotsReport(modelName, metaData, GANfolderPath)
  combinePDFs(GANfolderPath[1 : end - 1], modelName * " report"; leavePDFout = "report")
  writeGANmetaData(metaData; finalTxtPath = GANfolderPath)
  JLDfiles = readdir(GANfolderPath; join = true) |> x -> filter(y -> y[end - 4 : end] == ".jld2", x)
  metaData.genDefinition.neuralNetwork = cpu(metaData.genDefinition.neuralNetwork)
  metaData.discDefinition.neuralNetwork = cpu(metaData.discDefinition.neuralNetwork)
  save_object(GANfolderPath * string(length(JLDfiles) + 1) * ".jld2", metaData)
  plotGANlogs(readdir(GANfolderPath; join = true) |> x -> filter(y -> y[end - 4 : end] == ".jld2", x))
  mv(GANfolderPath * "logs/logPlots $(GANfolderPath[end - 4 : end - 1]).pdf",
    GANfolderPath * "logPlots $(GANfolderPath[end - 4 : end - 1]).pdf"
  )
  return nothing
end

# save both GAN NNs to BSON files
function saveGANs(metaData, currentEpoch; finalSave = false)
  # copy models to cpu
  cpuGenerator = metaData.genDefinition.neuralNetwork |> cpu
  cpuDiscriminator = metaData.discDefinition.neuralNetwork |> cpu
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
    metaData.genDefinition.neuralNetwork = gpu(cpuGenerator)
    metaData.discDefinition.neuralNetwork = gpu(cpuDiscriminator)
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
      savePath = datasetPath * "data/checkpoints/"
    else
      savePath = finalTxtPath
    end
  else # if running in colab
    savePath = "./gdrive/MyDrive/dataset files/GAN saves/"
  end
  open(
    prod([savePath, timeNow(), "metaData.txt"]),
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
    # write(id, "\n\nOPTIMISERS:" * "\n\tGENERATOR: " *
    #   printOptimizer(metaData.genDefinition.optInfo.opt) * " " *
    #   sciNotation(metaData.genDefinition.optInfo.opt.eta, 1) * "\n\tDISCRIMINATOR: " *
    #   printOptimizer(metaData.discDefinition.optInfo.opt) * " " *
    #   sciNotation(metaData.discDefinition.optInfo.opt.eta, 1) * "\n"
    # )
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
      write(id, "GENERATOR: " * sciNotation(metaData.lossesVals[:genTest][end], 3) * "\n")
      write(id, "DISCRIMINATOR: " * sciNotation(metaData.lossesVals[:discTest][end], 3) * "\n")
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