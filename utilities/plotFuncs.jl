# Functions to generate/save plots

# Use a trained model to predict samples and make plots comparing
# with ground truth. In the end, combine plots into single pdf file
function dispCNNtestPlots(quant::Int, path::String, dispTestLoader, finalName::String, FEparams, mlModel, lossFun)
  count = 0
  for (disp, force) in dispTestLoader # each batch
    for sampleInBatch in axes(disp)[end] # iterate inside batch
      count += 1; count > quant && break # limit number of tests
      plotDispTest( # plot comparison between prediction and truth
        FEparams, disp[:, :, :, sampleInBatch], Tuple([force[i][:, sampleInBatch]' for i in axes(force)[1]]),
        mlModel, 0, lossFun; folder = path, shiftForce = true)
    end
  end
  combinePDFs(path, finalName)
end

# Use a trained model to predict samples and make plots comparing
# with ground truth. In the end, combine plots into single pdf file.
# Adaptation for FEAloss pipeline
function dispCNNtestPlotsFEAloss(quant::Int, path::String, dispTestLoader, finalName::String, FEparams, mlModel, lossFun)
  count = 0
  for (disp, sup, vf, force) in dispTestLoader # each batch
    for sampleInBatch in axes(disp)[end] # iterate inside batch
      count += 1; count > quant && break # limit number of tests
      sampleDisp = dim4(disp[:, :, :, sampleInBatch])
      sampleForce = Tuple([force[i][:, sampleInBatch] for i in axes(force)[1]])
      plotDispTest( # plot comparison between prediction and truth
        FEparams, sampleDisp, sampleForce, mlModel, 0, lossFun; folder = path, shiftForce = true,
        FEAlossInput = (sampleDisp |> gpu |> mlModel |> cpu, sampleDisp, sup[:, :, sampleInBatch], [vf[sampleInBatch]], sampleForce))
    end
  end
  combinePDFs(path, finalName)
end

# Plot XAI results from each input channel for discriminator/critic
function explainDiscCritic(
  input::Array{Float32, 4}, analyzer::AbstractXAIMethod, network::String; goal = :display
)
  # get explanation for given input
  expl = analyzer(input)
  refs = Dict(
    :rows => [1, 1, 2, 2, 3], # columns in figure
    :cols => repeat([1, 2], 3), # columns in figure
    :channel => [1, 2, 4, 5, 7], # input channels to access
    # heatmap labels
    :label => ["VF", "VM", "Supports", "Forces", "Input topology"]
  )
  # reshaped input data
  data = dropdims.(eachslice(expl.attribution; dims = 3) |> collect .|> Array; dims = 3)
  imgWidth = 1100 # width of image
  fig = Figure(resolution = (imgWidth, 750)) # makie figure
  # columns and rows sizes
  colSize = round(Int, (0.95 * imgWidth) / 2); rowHeight = round(Int, colSize * (50/140))
  # iterate in input channels
  for (col, row, channel, label) in zip(refs[:cols], refs[:rows], refs[:channel], refs[:label])
    # axis, heatmap and colorbar for explanation of current input channel
    explAxis = Axis(fig[row, col]; height = rowHeight, title = label)
    explAxis.yreversed = true; hidespines!(explAxis); hidedecorations!(explAxis)
    heatmap!(explAxis, data[channel]' |> Array)
  end
  # fix size of columns
  [colsize!(fig.layout, i, Fixed(colSize)) for i in 1:2]
  # axis and heatmap of target topology
  topoAxis = Axis(fig[3, 2]; height = rowHeight, title = "Target topology")
  topoAxis.yreversed = true; hidespines!(topoAxis); hidedecorations!(topoAxis)
  heatmap!(topoAxis, input[:, :, 7, 1]' |> Array)
  if goal == :display
    GLMakie.activate!()
    display(fig)
  elseif goal == :save
    Makie.save("./networks/results/" * network * "-XAI.pdf", fig)
  end
end

# create plot with FEA inputs, and generated and real topologies.
function GANtestPlots(generator, dataPath, numSamples, savePath, modelName; extension = :png)
  # denseDataDict: compliance, vf, vm, energy, denseSupport, force, topology
  # dataDict_: compliance, vf, vm, energy, binarySupp, Fx, Fy, topologies
  denseDataDict, dataDict_ = denseInfoFromGANdataset(dataPath, numSamples)
  # load generator
  # generator, _ = loadGANs(generatorName, discriminatorName)
  for sample in 1:numSamples # loop in chosen samples
    # create makie figure and set it up
    fig = Figure(resolution = (1850, 750), fontsize = 19)
    colSize = 700; rowHeight = round(Int, colSize * (50/140))
    colsize!(fig.layout, 1, Fixed(colSize))
    # labels for first line of grid
    Label(fig[1, 1], "von Mises (MPa)")
    Label(fig[1, 3], "FEA input")
    colsize!(fig.layout, 3, Fixed(colSize))
    vmAxis = Axis(fig[2, 1]); vmAxis.yreversed = true
    vmHeatmap = heatmap!(vmAxis,denseDataDict[:vm][sample]' |> Array)
    Colorbar(fig[2, 2], vmHeatmap)
    FEAaxis = Axis(fig[2, 3]; height = rowHeight)
    limits!(FEAaxis, 1, FEAparams.meshSize[1], 1, FEAparams.meshSize[2])
    FEAaxis.yreversed = true
    heatmap!(FEAaxis, # plot supports
      suppToBinary(denseDataDict[:denseSupport][:, :, sample])[2]' |> Array
    )
    plotForce( # plot forces
      FEAparams, denseDataDict[:force][:, :, sample],
      fig, (2, 3), (2, 4); topologyGANtest = true,
      newAxis = FEAaxis, paintArrow = :orange,
      arrowWidth = 3, arrowHead = 20
    )
    Label(fig[3, 1],
      rpad("True topology", 29) * "VF = $(round(denseDataDict[:vf][sample]; digits = 3))" *
      lpad("Compliance", 23) * " = $(round(denseDataDict[:compliance][sample]; digits = 2)) N.mm"
    )
    trueTopoAxis, trueTopoHeatmap = heatmap(fig[4, 1],
      denseDataDict[:topologies][:, :, sample]' |> Array
    )
    Colorbar(fig[4, 2], trueTopoHeatmap)
    trueTopoAxis.yreversed = true; trueTopoAxis.height = rowHeight
    genInput = solidify( # concatenate input channels
      dataDict_[:vf][:, :, :, sample],
      dataDict_[:vm][:, :, :, sample],
      dataDict_[:energy][:, :, :, sample]
    )
    # optional data normalization in [-1; 1]
    normalizeDataset && (genInput = mapslices(normalizeVals, genInput; dims = [1 2]))
    fakeTopology = generator(genInput |> gpu) |> cpu
    # determine compliance of fake topology
    fakeCompliance = 0
    @suppress_err fakeCompliance = topologyCompliance(
      Float64(denseDataDict[:vf][sample]),
      Int.(denseDataDict[:denseSupport][:, :, sample]),
      Float64.(denseDataDict[:force][:, :, sample]),
      Float64.(fakeTopology[:, :, 1, 1])
    )
    Label(fig[3, 3],
      rpad("Fake topology", 29) * "VF = $(round(volFrac(fakeTopology)[1]; digits = 3))" *
      lpad("Compliance", 23) * " = $(round(fakeCompliance; digits = 2)) N.mm"
    )
    fakeTopoAxis, fakeTopoHeatmap = heatmap(fig[4, 3], fakeTopology[:, :, 1, 1]' |> Array)
    cb = Axis(fig[4, 4][1, 2]); hidespines!(cb); hidedecorations!(cb)
    Colorbar(fig[4, 4][1, 1], fakeTopoHeatmap)
    colsize!(fig.layout, 4, Fixed(150))
    fakeTopoAxis.yreversed = true
    if extension == :png
      GLMakie.activate!()
      Makie.save(savePath * "$(modelName)-$(sample).png", fig) # save pdf with plot
    elseif extension == :pdf
      CairoMakie.activate!()
      Makie.save(savePath * "$(modelName)-$(sample).pdf", fig) # save pdf with plot
    end
  end
end

# call GANtestPlots() for final model report
function GANtestPlotsReport(_modelName, _metaData, _path)
  GANtestPlots(
    # gpu(getGen(_metaData)), datasetPath * "data/test",
    gpu(getGen(_metaData)),
    readdir(datasetPath * "data/trainValidate"; join = true)[end],
    15, _path, _modelName; extension = :pdf
  )
end

# Define predict_function for calculations
ShapML.@everywhere function predict_function(model, data)
  topoBatch = model(cat(
    [solidify(data[sample, :]...) |> dim4 for sample in axes(data, 1)]...; dims = 4
  ))
  return DataFrame(:topo => [topoBatch[:, :, 1, s] for s in axes(topoBatch, 4)])
end

function genShapleyValPlots(split::Symbol, generator::Chain, nnName::String)
  # get input data
  _, genInput, _, _ = dataBatch(split, 1)
  inputFrame = DataFrame( # organize inside DataFrame
    :vf => [genInput[:, :, 1, s] for s in axes(genInput, 4)],
    :vm => [genInput[:, :, 2, s] for s in axes(genInput, 4)],
    :energy => [genInput[:, :, 3, s] for s in axes(genInput, 4)]
  )
  # more setup
  explain = copy(inputFrame[1:5, :])
  reference = copy(inputFrame[1:50, :])
  sample_size = 10
  # run simulation
  data_shap = ShapML.shap(explain = explain,
    reference = reference,
    model = cpu(generator),
    predict_function = predict_function,
    sample_size = sample_size,
    parallel = :samples,
    seed = 1
  )
  # plot heatmaps and save as pdf
  n = 0
  for row in eachrow(data_shap)
    for ele in row[3:end]
      n += 1
      imgWidth = 1300
      fig = Figure(resolution = (imgWidth, 700)) # makie figure
      # colSize = round(Int, (0.9 * imgWidth) / 4)
      # rowHeight = round(Int, colSize * (50/140))
      ax = Axis(fig[1, 1], title = "$n $(row[1]) $(row[2])")
      ax.yreversed = true
      hidespines!(ax); hidedecorations!(ax)
      hm = heatmap!(ax, ele' |> Array)
      Colorbar(fig[1, 2], hm)
      CairoMakie.activate!()
      Makie.save("./networks/results/topologyGANshap/1$n $(row[1]) $(row[2]).pdf", fig)
    end
  end
  combinePDFs("./networks/results/topologyGANshap/", "shaps")
end

# Use a trained model to predict samples and make plots comparing
# with ground truth. In the end, combine plots into single pdf file
function loadCNNtestPlots(
  quant::Int, path::String, vm::Array{Float32, 4}, forceData::Array{Float32, 3},
  finalName::String, FEparams, MLmodel, lossFun
)
  for sample in randDiffInt(quant, size(forceData, 3)) # loop in random samples
    plotVMtest(FEparams, vm[:, :, 1, sample], forceData[:, :, sample], MLmodel, 0, lossFun; folder = path)
  end
  combinePDFs(path, finalName)
end

function logsLines(value, name)
  f = Figure(resolution = (1500, 800)); # create makie figure
  ax = Axis(f[1:3, 1], xlabel = "Batches", # axis to draw on
    title = "Logged values",
  )
  # line plots of logs
  [lines!(ax, val; label = str) for (val, str) in zip(value, name)]
  # legend
  t = Axis(f[1, 2][1, 2]); hidespines!(t); hidedecorations!(t)
  Legend(f[1, 2][1, 1], ax)
  colsize!(f.layout, 2, Fixed(160))
  Makie.save(GANfolderPath * "logs/$(rand(1:9999)).pdf", f) # save pdf
end

# Generate pdf with list of hyperparameters used to train model.
# To be used in hyperGridSave
function parameterList(model, opt, lossFun, path; multiLossArch = false)
  # create makie figure and set it up
  fig = Figure(resolution = (1400, 1000), fontsize = 20);
  currentLine = 1 # count lines
  labelHeight = 10
  Label(fig[currentLine, 1], "NN Architecture:"; height = labelHeight)
  # Strings describing each layer
  if multiLossArch
    layerString = [string(model.layers[1].layers[l]) for l in 1:length(model.layers[1].layers)]
    layerString = vcat(layerString, [string(model.layers[2].paths[p]) for p in 1:length(model.layers[2].paths)])
  else
    layerString = [string(model.layers[m].layers[l]) for m in 1:length(model.layers) for l in 1:length(model.layers[m].layers)]
  end
  # Individually include layer strings as labels
  for line in axes(layerString)[1]
    Label(fig[currentLine, 2], layerString[line]; height = labelHeight)
    currentLine += 1 # Update current line
  end
  Label(fig[currentLine, 1], "Optimizer:"; height = labelHeight) # optimizer info
  Label(fig[currentLine, 2], string(typeof(opt)); height = labelHeight)
  currentLine += 1 # Update current line
  Label(fig[currentLine, 2], sciNotation(opt.eta, 0); height = labelHeight)
  currentLine += 1 # Update current line
  Label(fig[currentLine, 1], "Loss function"; height = labelHeight) # loss function used
  Label(fig[currentLine, 2], string(lossFun); height = labelHeight)
  currentLine += 1 # Update current line
  Label(fig[currentLine, 1], "# parameters:"; height = labelHeight) # number of model parameters
  Label(fig[currentLine, 2], string(sum(length, Flux.params(model))); height = labelHeight)
  Makie.save("$path/$(rand(1:999999)).pdf", fig)
end

# Make plot for disp model test and visually compare ML model predictions against truth
function plotDispTest(
  FEparams, disp, trueForces, model, modelName, lossFun;
  folder = "", maxDenormalize = 0, shiftForce = false, FEAlossInput = 0
)
  if FEAlossInput == 0
    # Reshape output of loadCNN to match dataset format
    predForces = convert.(Float32, dim4(disp)) |> gpu |> model |> cpu
    trueForces = Tuple([permutedims(trueForces[i], (2, 1)) for i in axes(trueForces)[1]])
  else
    predForces = FEAlossInput[1]
  end
  if shiftForce # shift force components back to [-90; 90] range if necessary
    [predForces[i] .-= 90 for i in 3:4]
    [trueForces[i] .-= 90 for i in 3:4]
  end
  maxDenormalize != 0 && (predForces *= maxDenormalize) # Denormalize force vectors if necessary
  fig = Figure(resolution = (1000, 700), fontsize = 20) # create makie figure and set it up
  axHeight = 200 # axis height for vm and forces
  textPos = (-0.1, 0.5) # text position in final figure
  # plot true forces
  forceAxis = plotForce(FEparams, trueForces, fig, (1, 2:3), (2, 2); alignText = textPos, axisHeight = axHeight)
  # plot predicted forces
  plotForce(FEparams, predForces, fig, (1, 2:3), (2, 3);
    newAxis = forceAxis, paintArrow = :green, paintText = :green, alignText = textPos)
  # Labels
  Label(fig[1, 1], "Force positions"; tellheight = :false)
  (colsize!(fig.layout, i, Fixed(500)) for i in 1:2)
  # loss value of current prediction
  if FEAlossInput == 0
    Label(fig[2, 1], "Loss: "*sciNotation(lossFun(predForces, trueForces), 3); align = (-1, 0.5), tellheight = :false)
  else
    Label(fig[2, 1], "Loss: "*sciNotation(lossFun(FEAlossInput...), 3); align = (-1, 0.5), tellheight = :false)
  end
  if length(folder) == 0 # save figure created
    Makie.save("./networks/trainingPlots/$modelName test.pdf", fig)
  else
    Makie.save("$folder/$(rand(1:999999)).pdf", fig)
  end
end

# plot forces
function plotForce(
  FEAparams, forces, fig, arrowsPos, textPos;
  newAxis = "true", paintArrow = :black, paintText = :black,
  alignText = (:left, :center), axisHeight = 0,
  topologyGANtest = false, arrowWidth = 1.5,
  arrowHead = 1.5, includeText = True
)
  if typeof(forces) <: Tuple
    forceMat = reduce(hcat, [forces[i] for i in axes(forces)[1]])
  else
    forceMat = forces
  end
  loadXcoord = zeros(size(forceMat, 1)); loadYcoord = zeros(size(forceMat, 1))
  for l in axes(forceMat, 1) # Get load positions
    loadXcoord[l] = forceMat[l, 2]
    loadYcoord[l] = forceMat[l, 1]
  end
  if typeof(newAxis) == String # create and setup plotting axis
    if axisHeight == 0
      axis = Axis(fig[arrowsPos[1], arrowsPos[2]])
    else
      axis = Axis(fig[arrowsPos[1], arrowsPos[2]]; height = axisHeight)
    end
    if !topologyGANtest
      limits!(axis,
        -round(0.03 * FEAparams.meshSize[1]), round(1.03 * FEAparams.meshSize[1]),
        -round(0.1 * FEAparams.meshSize[2]), round(1.1 * FEAparams.meshSize[2])
      )
      hlines!(axis,
        [0, FEAparams.meshSize[2]], xmin = [0.0, 0.0],
        xmax = [FEAparams.meshSize[1], FEAparams.meshSize[1]], color = :black
      )
      vlines!(axis,
        [0, FEAparams.meshSize[1]], ymin = [0.0, 0.0],
        ymax = [FEAparams.meshSize[2], FEAparams.meshSize[2]], color = :black
      )
    end
  else
    axis = newAxis
  end
  arrows!( # Plot loads as arrows
    axis, loadXcoord, loadYcoord, forceMat[:, 3], -forceMat[:, 4];
    # norm of weakest force. Used to scale force vectors
    lengthscale = 1 / (0.1 * minimum( sqrt.( (forceMat[:, 3]) .^ 2 + (forceMat[:, 4]) .^ 2 ) )),
    linecolor = paintArrow, arrowcolor = paintArrow,
    linewidth = arrowWidth, arrowsize = arrowHead
  )
  if includeText
    text!(
      loadXcoord, loadYcoord,
      text = ["1", "2"], color = :white
    )
    # text with values of force components
    f1 = "Forces (N):\n1: $(round(Int, forceMat[1, 3])); $(round(Int, forceMat[1, 4]))\n"
    f2 = "2: $(round(Int, forceMat[2, 3])); $(round(Int, forceMat[2, 4]))"
    t = Axis(fig[textPos[1], textPos[2]]); hidespines!(t); hidedecorations!(t)
    text!(t,
        f1*f2; color = paintText,
        align = alignText, offset = (-60, 0)
    )
  end
  return axis
end

# line plots of histories of GAN losses
function plotGANlogs(JLDpath::Vector{String})
  mkpath(GANfolderPath * "logs") # folder for PDFs with plots
  CairoMakie.activate!()
  if haskey(load(JLDpath[1])["single_stored_object"].discDefinition.nnValues, :discTrue)
    foolDisc, mse, vfMAE, discTrue, discFalse = Vector{Float32}[], Vector{Float32}[], Vector{Float32}[], Vector{Float32}[], Vector{Float32}[]
    for filePath in JLDpath
      savedStruct = 0
      @suppress_err savedStruct = load(filePath)["single_stored_object"]
      push!(foolDisc, get(savedStruct.genDefinition.nnValues[:foolDisc])[2][2 : end])
      push!(mse, get(savedStruct.genDefinition.nnValues[:mse])[2][2 : end])
      push!(vfMAE, get(savedStruct.genDefinition.nnValues[:vfMAE])[2][2 : end])
      push!(discTrue, get(savedStruct.discDefinition.nnValues[:discTrue])[2][2 : end])
      push!(discFalse, get(savedStruct.discDefinition.nnValues[:discFalse])[2][2 : end])
    end
    foolDisc = foolDisc |> Iterators.flatten |> collect
    mse = mse |> Iterators.flatten |> collect
    vfMAE = vfMAE |> Iterators.flatten |> collect
    discTrue = discTrue |> Iterators.flatten |> collect
    discFalse = discFalse |> Iterators.flatten |> collect
    logsLines( # all intermediate terms
      [foolDisc, mse, vfMAE, discTrue, discFalse],
      ["foolDisc", "mse", "vfMAE", "discTrue", "discFalse"]
    )
    logsLines( # both final losses
      [foolDisc + mse + vfMAE, discTrue + discFalse],
      ["generator loss", "discriminator loss"]
    )
    logsLines( # intermediate and final generator values
      [foolDisc, mse, vfMAE, foolDisc + mse + vfMAE],
      ["foolDisc", "mse", "vfMAE", "generator loss"]
    )
    logsLines( # intermediate and final discriminator values
      [discTrue, discFalse, discTrue + discFalse],
      ["discTrue", "discFalse", "discriminator loss"]
    )
  else # wgan was used
    criticOutFake, criticOutReal, genDoutFake, mse = Vector{Float32}[], Vector{Float32}[], Vector{Float32}[], Vector{Float32}[]
    for filePath in JLDpath
      savedStruct = 0
      @suppress_err savedStruct = load(filePath)["single_stored_object"]
      push!(criticOutFake, get(savedStruct.discDefinition.nnValues[:criticOutFake])[2][2 : end])
      push!(criticOutReal, get(savedStruct.discDefinition.nnValues[:criticOutReal])[2][2 : end])
      push!(genDoutFake, get(savedStruct.discDefinition.nnValues[:genDoutFake])[2][2 : end])
      push!(mse, get(savedStruct.discDefinition.nnValues[:mse])[2][2 : end])
    end
    criticOutFake = criticOutFake |> Iterators.flatten |> collect
    criticOutReal = criticOutReal |> Iterators.flatten |> collect
    genDoutFake = genDoutFake |> Iterators.flatten |> collect
    mse = mse |> Iterators.flatten |> collect
    logsLines([criticOutFake - criticOutReal], ["critic loss"])
    logsLines([genDoutFake + mse], ["generator loss"])
    logsLines([mse], ["mse"])
    logsLines([genDoutFake], ["genDoutFake"])
    logsLines([criticOutFake],["discFake"])
    logsLines([criticOutReal], ["discReal"])
  end
  combinePDFs(GANfolderPath * "logs", "logPlots $(GANfolderPath[end - 4 : end - 1])")
end

# create line plots of GAN validation histories.
# save plot as pdf
function plotGANValHist(lossesVals, validFreq, modelName; metaDataName = " ", midTraining = false)
  if metaDataName != " "
    # get values from saved txt file
    genValHistory, discValHistory, testLosses, validFreq = getValuesFromTxt(metaDataName)
  else # get values from GANmetaData struct
    genValHistory = lossesVals[:genValHistory]
    discValHistory = lossesVals[:discValHistory]
    if midTraining
      testLosses = (0.0, 0.0)
    else
      testLosses = (lossesVals[:genTest][end], lossesVals[:discTest][end])
    end
  end
  # GLMakie.activate!() # rasterization
  # maxima and minima of validation histories
  minima = findmin.((genValHistory, discValHistory))
  f = Figure(resolution = (1500, 800)); # create makie figure
  ax = Axis(f[1:3, 1], yscale = identity, # axis to draw on
    xlabel = "Epochs", ylabel = "Losses", title = modelName
  )
  # set limits of x axis
  xlims!(ax, 0, validFreq * (length(genValHistory) + 1))
  # line plots of histories
  lineGen = lines!(ax, validFreq:validFreq:validFreq * length(genValHistory), genValHistory)
  lineDisc = lines!(ax, validFreq:validFreq:validFreq * length(discValHistory), discValHistory)
  # add legend to plot
  t = Axis(f[1, 2][1, 2]); hidespines!(t); hidedecorations!(t)
  Legend(f[1, 2][1, 1], [lineGen, lineDisc], ["Generator", "Discriminator"])
  colsize!(f.layout, 2, Fixed(350))
  minimaAxis = Axis(f[2, 2])
  text!( # text for validation minima for both NNs
    minimaAxis, "Validation minima:\n\n" * lpad("Epoch:", 29) * lpad("Value:", 9) * "\n" * 
    rpad("Generator:", 16) * rpad(minima[1][2] * validFreq, 12) * (sciNotation <| (minima[1][1], 3)...) * "\n" *
    "Discriminator: " * rpad(minima[2][2] * validFreq, 12) * (sciNotation <| (minima[2][1], 3)...),
    offset = (-170, -30),
  )
  hidespines!(minimaAxis); hidedecorations!(minimaAxis)
  testAxis = Axis(f[3, 2])
  text!( # text for test loss values
    testAxis, "Test losses:\n\n" * 
    rpad("Generator:", 16) * (sciNotation <| (testLosses[1], 3)...) * "\n" *
    "Discriminator: " * (sciNotation <| (testLosses[2], 3)...),
    offset = (-170, 0),
  )
  hidespines!(testAxis); hidedecorations!(testAxis)
  if midTraining
    GLMakie.activate!()
    display(f)
  else
    CairoMakie.activate!() # vector graphics
    Makie.save(GANfolderPath * "validation histories.pdf", f) # save pdf with plot
  end
end

# Line plots of evaluation histories
function plotLearnTries(trainParams, tries; drawLegend = true, name = timeNow(), path = "./networks/trainingPlots")
  f = Figure(resolution = (1050, 700)); # create makie figure
  ax = Axis(f[1:2, 1], xlabel = "Validation epochs", ylabel = "Loss", title = name)
  colsize!(f.layout, 1, Fixed(600))
  # plot evaluation histories of all runs
  runs = [lines!(convert.(Float32, trainParams[run].evaluations)) for run in 1:length(tries)]
  # if learning rate decay was used, mark epochs with vertical lines
  if trainParams[1].schedule != 0
    decayPerValidation = ceil(Int, trainParams[1].schedule/trainParams[1].validFreq)
    vlines!(ax, decayPerValidation:decayPerValidation:length(trainParams[1].evaluations), color = :darkslategrey, linestyle = :dash)
  end
  drawLegend && Legend(f[2, 2], runs, string.(tries)) # include legend
  minimaText = [] # label with evaluation loss minimum e corresponding epoch
  for i in axes(trainParams)[1]
    valMin = findmin(trainParams[i].evaluations)
    minimaText = vcat(minimaText, "$i) Min. Loss: $(sciNotation(valMin[1], 3))   Epoch: $(valMin[2])")
  end
  if length(trainParams) > 1
    minimaText[1:end-1] .= minimaText[1:end-1].*["\n" for i in 1:length(minimaText) - 1]
  else
    minimaText[1] = minimaText[1]*"\n"
  end
  t = text(f[1,2], prod(minimaText); fontsize = 15, align = (0.5, 0.0))
  textConfig(t) # setup label
  colsize!(f.layout, 2, Fixed(300))
  Makie.save("$path/$name.pdf", f) # save pdf with plot
end

# create figure to vizualize sample
function plotSample(FEAparams, supps, forces, vf, top, disp, dataset, section, sample; goal = "display")
  # create makie figure and set it up
  fig = Figure(resolution = (1400, 700))
  colSize = 500
  colsize!(fig.layout, 1, Fixed(colSize))
  # labels for first line of grid
  Label(fig[1, 1], "Supports", fontsize = 20)
  Label(fig[1, 2], "Force positions", fontsize = 20)
  colsize!(fig.layout, 2, Fixed(colSize))
  # plot supports
  plotSupps(FEAparams, supps, fig)
  # plot forces
  plotForce(FEAparams, forces, fig, (2, 2), (2, 3))
  # labels for second line of grid
  Label(fig[3, 1], "Topology VF = $(round(vf; digits = 3))", fontsize = 20)
  Label(fig[3, 2], "von Mises (MPa)", fontsize = 20)
  # plot final topology
  heatmap(fig[4, 1], 1:FEAparams.meshSize[2], FEAparams.meshSize[1]:-1:1, top')
  # plot von mises field
  plotVM(FEAparams, disp, vf, fig, (4, 2))
  if goal == "save"
    # save image file
    Makie.save(datasetPath*"analyses/isolated features/imgs/$dataset $section $sample.pdf", fig)
  elseif goal == "display"
    GLMakie.activate!()
    display(fig)
  else
    error("\n\nWrong goal for plotSample().\n\n")
  end
end

# include plot of supports
function plotSupps(FEAparams, supp, fig; loadAxis = 0)
  _ , _suppMatElement = suppToBinary(supp)
  # plot supports
  if loadAxis == 0
    heatmap(fig[2, 1],
      1:FEAparams.meshSize[2], FEAparams.meshSize[1]:-1:1,
      _suppMatElement'
    )
  else
    heatmap!(loadAxis,
      1:FEAparams.meshSize[2],
      FEAparams.meshSize[1]:-1:1, _suppMatElement'
    )
  end
end

# test bounds for detecting topologies with too many intermediate densities
function plotTopoIntermediate(forces, supps, vf, top, FEAparams, bound)
  
  quantForces = 2
  colSize = 500
  count = 0
  count += 1
  # create makie figure and set it up
  fig = Figure(resolution = (1400, 700));
  colsize!(fig.layout, 1, Fixed(colSize))
  # labels for first line of grid
  Label(fig[1, 1], "Supports", fontsize = 20)
  Label(fig[1, 2], "Force positions", fontsize = 20)
  colsize!(fig.layout, 2, Fixed(colSize))
  supports = zeros(FEAparams.meshSize)'
  if supps[1,3] > 3


    if supps[1,3] == 4
      # left
      supports[:,1] .= 3
    elseif supps[1,3] == 5
      # bottom
      supports[end,:] .= 3
    elseif supps[1,3] == 6
      # right
      supports[:,end] .= 3
    elseif supps[1,3] == 7
      # top
      supports[1,:] .= 3
    end


  else

    [supports[supps[m, 1], supps[m, 2]] = 3 for m in 1:size(supps)[1]]

  end
  # plot supports
  heatmap(fig[2,1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,supports')
  # plot forces
  loadXcoord = zeros(quantForces)
  loadYcoord = zeros(quantForces)
  for l in 1:quantForces
    loadXcoord[l] = forces[l,2]
    loadYcoord[l] = FEAparams.meshSize[2] - forces[l,1] + 1
  end
  axis = Axis(fig[2,2])
  xlims!(axis, -round(0.1*FEAparams.meshSize[1]), round(1.1*FEAparams.meshSize[1]))
  ylims!(axis, -round(0.1*FEAparams.meshSize[2]), round(1.1*FEAparams.meshSize[2]))
  arrows!(
    axis, loadXcoord, loadYcoord,
    forces[:,3], forces[:,4];
    # norm of weakest force. will be used to scale force vectors in arrows!() command
    lengthscale = 1 / (0.1*minimum( sqrt.( (forces[:,3]).^2 + (forces[:,4]).^2 ) ))
  )
  # quantity of elements with densities between 0.5 +/- bound
  intermQuant = length(filter(
    x -> (x > 0.5 - bound) && (x < 0.5 + bound),
    reshape(top, (1,:))
  ))
  intermPercent = "$( round(intermQuant/length(top)*100;digits=2) )%"
  l = text(
      fig[4,2],
      intermPercent;
  )
  textConfig(l)
  Label(fig[3, 1], "Topology VF = $(round(vf;digits=3))", fontsize = 20) # labels for second line of grid
  heatmap(fig[4, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,top') # plot final topology
  save(datasetPath*"analyses/intermediateDensities/$(rand(1:99999)).pdf", fig) # save image file
end

function plotTopoPred(targetTopo, predTopo; goal = "save")
  fig = Figure(resolution = (1100, 800)) # create makie figure
  # labels for each heatmap
  Label(fig[1, 1], "Original", fontsize = 20; tellheight = false); Label(fig[2, 1], "Prediction", fontsize = 20; tellheight = false)
  _, hmPred = heatmap(fig[2, 2], # heatmap of predicted topology
    Array(reshape(predTopo, (FEAparams.meshSize[2], FEAparams.meshSize[1]))')
  )
  colsize!(fig.layout, 2, Fixed(800))
  # heatmap of original topology
  _, hmOriginal = heatmap(fig[1, 2], Array(targetTopo'))
  # colorbars
  Colorbar(fig[1, 3], hmPred); Colorbar(fig[2, 3], hmOriginal)
  if goal == "display" # display image
    GLMakie.activate!()
    display(fig)
  elseif goal == "save" # save image
    CairoMakie.activate!()
    Makie.save("./networks/test.pdf", fig)
  else
    error("Invalid kw arg 'goal' in plotTopoPred().")
  end
end

# plot von Mises field
function plotVM(FEAparams, disp, vf, fig, figPos)
  # get von Mises field
  vm = calcVM(FEAparams.nElements, FEAparams, disp, 210e3*vf, 0.3)
  # plot von Mises
  _,hm = heatmap(fig[figPos[1], figPos[2]],
    1:FEAparams.meshSize[2], FEAparams.meshSize[1]:-1:1, vm';
    colorrange = (floor(Int, minimum(vm)), ceil(Int, maximum(vm)))
  )
  Colorbar(fig[figPos[1], figPos[2] + 1], hm)
end

# Make plot for VM model test and visually compare ML model predictions against truth
function plotVMtest(FEparams, input, trueForces, model, modelName, lossFun; folder = "", maxDenormalize = 0)
  # Reshape output of loadCNN to match dataset format
  if size(input, 3) == 1
    predForces = convert.(Float32, reshape(input, (size(input)..., 1, 1))) |> gpu |> model |> cpu
  elseif size(input, 3) == 2
    predForces = convert.(Float32, reshape(input, (size(input)..., 1, 1))) |> gpu |> model |> cpu
  end
  maxDenormalize != 0 && (predForces *= maxDenormalize) # Denormalize force vectors if necessary
  fig = Figure(resolution = (1000, 700), fontsize = 20) # create makie figure and set it up
  axHeight = 200 # axis height for vm and forces
  vmAxis = Axis(fig[1, 2:3]; height = axHeight)
  heatmap!(vmAxis, 1:FEparams.meshSize[1], FEparams.meshSize[2]:-1:1, vm') # plot vm
  textPos = (-0.1, 0.5) # text position in final figure
  forceAxis = plotForce(FEparams, trueForces, fig, (2, 2:3), (3, 2); alignText = textPos, axisHeight = axHeight) # plot true forces
  # plot predicted forces
  plotForce(FEparams, reshapeForces(predForces), fig, (2, 2:3), (3, 3); newAxis = forceAxis, paintArrow = :green, paintText = :green, alignText = textPos)
  # Labels
  Label(fig[1, 1], "von Mises field"; tellheight = :false)
  Label(fig[2, 1], "Force positions"; tellheight = :false)
  (colsize!(fig.layout, i, Fixed(500)) for i in 1:2)
  # loss value of current prediction
  Label(fig[3, 1], "Loss: "*sciNotation(lossFun(predForces, reshape(trueForces, (1, :))'), 3);
  align = (-1, 0.5), tellheight = :false)
  if length(folder) == 0 # save figure created
    Makie.save("./networks/trainingPlots/$modelName test.pdf", fig)
  else
    Makie.save("$folder/$(rand(1:999999)).pdf", fig)
  end
end

# Generate pdf with tests using model that was trained on normalized data
function testNormalizedData(modelPath::String, vm, forceData, quant::Int64, FEparams, lossFun)
  files = glob("*", modelPath)
  BSON.@load filter(x -> x[end-3:end] == "bson", files) cpu_model
  mkpath(modelPath*"/normalizedTests")
  for sample in randDiffInt(quant, size(forceData, 3))
    plotVMtest(
      FEparams, vm[:, :, 1, sample], forceData[:, :, sample], gpu(cpu_model), 0, lossFun;
      folder = modelPath, maxDenormalize = maxForceMat)
  end
  combinePDFs(path, "finalName")
end

# Setup text element for plotting
function textConfig(l)
  l.axis.attributes.xgridvisible = false
  l.axis.attributes.ygridvisible = false
  l.axis.attributes.rightspinevisible = false
  l.axis.attributes.leftspinevisible = false
  l.axis.attributes.topspinevisible = false
  l.axis.attributes.bottomspinevisible = false
  l.axis.attributes.xticksvisible = false
  l.axis.attributes.yticksvisible = false
  l.axis.attributes.xlabelvisible = false
  l.axis.attributes.ylabelvisible = false
  l.axis.attributes.titlevisible  = false
  l.axis.attributes.xticklabelsvisible = false
  l.axis.attributes.yticklabelsvisible = false
  l.axis.attributes.tellheight = false
  l.axis.attributes.tellwidth = false
  l.axis.attributes.halign = :center
  l.axis.attributes.width = 300
end

# plot compliance, VF and topology errors from
# trained models in all 3 data splits
function trainedPerformancePlot(
  topoGANerror::Dict{Symbol, Vector{Float32}},
  convNextError::Dict{Symbol, Vector{Float32}}
)
  fig = Figure(resolution = (1200, 700)) # makie figure
  topoGANseAx = Axis(fig[1, 1])
  topoGANrelAx = Axis(fig[1, 2])
  convNextSEax = Axis(fig[2, 1])
  convNextRelAx = Axis(fig[2, 2])
  bin = 20
  ### topologyGAN histograms
  # U-SE-ResNet topology squared error
  hist!(topoGANseAx, topoGANerror[:topoSE]; bins = bin,
    color = RGBA{Float64}(0.48, 0.41, 0.93,0.6), normalization = :probability
  )
  # U-SE-ResNet relative VF error
  hist!(topoGANrelAx, topoGANerror[:VFerror]; bins = bin,
    color = RGBA{Float64}(0.0, 0.8, 0.8, 0.6), normalization = :probability
  )
  ## QuickTO histograms
  # ConvNeXt topology squared error
  hist!(convNextSEax, convNextError[:topoSE]; bins = bin,
    color = RGBA{Float64}(0.48, 0.41, 0.93,0.6), normalization = :probability
  )
  # ConvNeXt relative VF error
  hist!(convNextRelAx, convNextError[:VFerror]; bins = bin,
    color = RGBA{Float64}(0.0, 0.8, 0.8, 0.6), normalization = :probability
  )
  # GLMakie.activate!()
  # display(fig)
  CairoMakie.activate!()
  Makie.save(projPath * "networks/results/TrainedTestErrorMetrics.pdf", fig)
end

# using trained models, create plot including
# generator input, and generated and real topologies
function trainedSamples(
  imageAmount::Int, samplesPerImage::Int, gen::Chain, NNtype::String;
  goal = :save, split = :training
)
  sampleAmount = imageAmount * samplesPerImage
  if split == :training # file used in training
    sampleSource = readdir(datasetPath * "data/trainValidate"; join = true)[1]
    println(sampleSource)
  elseif split == :validation # file used in validation, but still varied
    sampleSource = readdir(datasetPath * "data/trainValidate"; join = true)[end]
    println(sampleSource)
  elseif split == :test # file with unseen type of support
    sampleSource = datasetPath * "data/test"
  else
    error("Wrong 'split' kwarg in 'trainedSamples()'.")
  end
  randID = 0
  # denseDataDict: compliance, vf, vm, energy, denseSupport, force, topology
  # MLdataDict: compliance, vf, vm, energy, binarySupp, Fx, Fy, topologies
  denseDataDict, MLdataDict = denseInfoFromGANdataset(sampleSource, sampleAmount)
  # tensor initializers
  genInput = zeros(Float32, FEAparams.meshMatrixSize..., 3, 1); FEAinfo = similar(genInput)
  topology = zeros(Float32, FEAparams.meshMatrixSize..., 1, 1)
  genInput, FEAinfo, topology = groupGANdata!(
    genInput, FEAinfo, topology, MLdataDict; sampleAmount = 0
  )
  # discard first position of arrays (initialization)
  genInput, _, _ = remFirstSample.((genInput, FEAinfo, topology))
  dataBatch = DataLoader(genInput; batchsize = samplesPerImage, parallel = true)
  # each image
  imgWidth = 1300
  fig = Figure(resolution = (imgWidth, 700)) # makie figure
  colSize = round(Int, (0.9 * imgWidth) / 4); rowHeight = round(Int, colSize * (50/140))
  Label(fig[1, 1], "FEM inputs"; lineheight = 0.6)
  Label(fig[1, 2], "von Mises")
  Label(fig[1, 3], "Fake topology"); Label(fig[1, 4], "Real topology")
  [colsize!(fig.layout, i, Fixed(colSize)) for i in 1:4]
  for (batchIndex, gIn) in enumerate(dataBatch)
    fakeTopology = gen(gIn |> gpu) |> cpu |> padGen # use generator
    for sample in 1:samplesPerImage # each sample in current image
      sampleID = samplesPerImage * (batchIndex - 1) + sample
      ## FEA intputs (supports and loads)
      FEAaxis = Axis(fig[sample + 1, 1]; height = rowHeight)
      hidespines!(FEAaxis); hidedecorations!(FEAaxis)
      limits!(FEAaxis, 1, FEAparams.meshSize[1], 1, FEAparams.meshSize[2])
      FEAaxis.yreversed = true
      heatmap!(FEAaxis, # plot supports
        suppToBinary(denseDataDict[:denseSupport][:, :, sampleID])[2]' |> Array,
        colormap = :bluesreds
      )
      plotForce( # plot forces
        FEAparams, denseDataDict[:force][:, :, sampleID],
        fig, (2, 3), (2, 4); topologyGANtest = true,
        newAxis = FEAaxis, paintArrow = :orange,
        arrowWidth = 2, arrowHead = 15, includeText = false
      )
      ## VM
      vmAxis = Axis(fig[sample + 1, 2]; height = rowHeight)
      vmAxis.yreversed = true; hidespines!(vmAxis); hidedecorations!(vmAxis)
      heatmap!(vmAxis, denseDataDict[:vm][sampleID]' |> Array)
      ## fakeTopology
      fakeTopoAxis = Axis(fig[sample + 1, 3]; height = rowHeight)
      fakeTopoAxis.yreversed = true
      heatmap!(fakeTopoAxis, fakeTopology[:, :, 1, sample]' |> Array)
      hidespines!(fakeTopoAxis); hidedecorations!(fakeTopoAxis)
      ## true topology
      trueTopoAxis = Axis(fig[sample + 1, 4]; height = rowHeight)
      heatmap!(trueTopoAxis,
        denseDataDict[:topologies][:, :, sampleID]' |> Array
      )
      hidespines!(trueTopoAxis); hidedecorations!(trueTopoAxis)
      trueTopoAxis.yreversed = true
    end
    if goal == :save # save image file as pdf
      CairoMakie.activate!()
      if batchIndex == 1
        randID = rand(1:9999)
        mkdir(
          "./networks/results/$NNtype $split $randID"
        )
      end
      Makie.save(projPath * "networks/results/$NNtype $split $randID/$batchIndex.pdf", fig)
    elseif goal == :display # plot batch and exit
      GLMakie.activate!()
      display(fig)
      return Nothing
    end
  end
end