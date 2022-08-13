# Functions to generate/save plots

# Generate pdf with validation history and test plots for hyperparameter grid search
function hyperGridSave(currentModelName, trainParams, tries, vm, forceData, FEparams, MLmodel)
  # Create folder for current model
  mkpath("./networks/models/$currentModelName")
  plotLearnTries([trainParams], [tries]; drawLegend = false, name = currentModelName*"valLoss", path = "./networks/models/$currentModelName")
  # Create test plots and unite all PDF files in folder
  stressCNNtestPlots(10, "./networks/models/$currentModelName", vm, forceData, currentModelName, FEparams, MLmodel)
end

# plot forces
function plotForce(
  FEAparams, forces, fig, arrowsPos, textPos;
  newAxis = "true", paintArrow = :black, paintText = :black, alignText = (0, 0), axisHeight = 0
)
  loadXcoord = zeros(size(forces, 1))
  loadYcoord = zeros(size(forces, 1))
  # Get loads positions
  for l in 1:size(forces, 1)
    loadXcoord[l] = forces[l,2]
    loadYcoord[l] = FEAparams.meshSize[2] - forces[l,1] + 1
  end
  # create and setup plotting axis
  if typeof(newAxis) == String
    if axisHeight == 0
      axis = Axis(fig[arrowsPos[1], arrowsPos[2]])
    else
      axis = Axis(fig[arrowsPos[1], arrowsPos[2]]; height = axisHeight)
    end
    xlims!(axis, -round(0.03*FEAparams.meshSize[1]), round(1.03*FEAparams.meshSize[1]))
    ylims!(axis, -round(0.1*FEAparams.meshSize[2]), round(1.1*FEAparams.meshSize[2]))
    hlines!(axis, [0, FEAparams.meshSize[2]], xmin = [0.0, 0.0], xmax = [FEAparams.meshSize[1], FEAparams.meshSize[1]], color = :black)
    vlines!(axis, [0, FEAparams.meshSize[1]], ymin = [0.0, 0.0], ymax = [FEAparams.meshSize[2], FEAparams.meshSize[2]], color = :black)
  else
    axis = newAxis
  end
  # Plot loads as arrows
  arrows!(
    axis, loadXcoord, loadYcoord,
    forces[:,3], forces[:,4];
    # norm of weakest force. will be used to scale force vectors in arrows!() command
    lengthscale = 1 / (0.1*minimum( sqrt.( (forces[:,3]).^2 + (forces[:,4]).^2 ) )),
    linecolor = paintArrow,
    arrowcolor = paintArrow,
  )
  # text with values of force components
  f1 = "Forces (N):\n1: $(round(Int,forces[1, 3])); $(round(Int,forces[1, 4]))\n"
  f2 = "2: $(round(Int,forces[2, 3])); $(round(Int,forces[2, 4]))"
  l = text(
      fig[textPos[1], textPos[2]],
      f1*f2;
      color = paintText,
      align = alignText
  )
  textConfig(l)
  return axis
end

# Line plots of evaluation histories
function plotLearnTries(trainParams, tries; drawLegend = true, name = timeNow(), path = "./networks/trainingPlots")
  f = Figure(resolution = (1050, 700));
  ax = Axis(f[1:2, 1], xlabel = "Validation epochs", ylabel = "Loss", title = name)
  colsize!(f.layout, 1, Fixed(600))
  runs = [lines!(convert.(Float32, trainParams[run].evaluations)) for run in 1:length(tries)]
  if trainParams[1].schedule != 0
    decayPerValidation = trainParams[1].schedule/trainParams[1].evalFreq
    vlines!(ax, decayPerValidation:decayPerValidation:length(trainParams[1].evaluations))
  end
  drawLegend && Legend(f[2, 2], runs, string.(tries))
  minimaText = []
  for i in 1:length(trainParams)
    valMin = findmin(trainParams[i].evaluations)
    minimaText = vcat(
      minimaText, "$i) Min. Loss: $(sciNotation(valMin[1], 3))   Epoch: $(valMin[2])"
      )
  end
  if length(trainParams) > 1
    minimaText[1:end-1] .= minimaText[1:end-1].*["\n" for i in 1:length(minimaText) - 1]
  else
    minimaText[1] = minimaText[1]*"\n"
  end
  t = text(f[1,2], prod(minimaText); textsize = 15, align = (0.5, 0.0))
  textConfig(t)
  colsize!(f.layout, 2, Fixed(300))
  Makie.save("$path/$name.pdf", f)
end

# create figure to vizualize sample
function plotSample(FEAparams, supps, forces, vf, top, disp, dataset, section, sample; goal = "display")
  # create makie figure and set it up
  fig = Figure(resolution = (1400, 700));
  colSize = 500
  colsize!(fig.layout, 1, Fixed(colSize))
  # labels for first line of grid
  Label(fig[1, 1], "Supports", textsize = 20)
  Label(fig[1, 2], "Force positions", textsize = 20)
  colsize!(fig.layout, 2, Fixed(colSize))
  # plot supports
  plotSupps(FEAparams, supps, fig)
  # plot forces
  plotForce(FEAparams, forces, fig, (2, 2), (2, 3))
  # labels for second line of grid
  Label(fig[3, 1], "Topology VF = $(round(vf; digits = 3))", textsize = 20)
  Label(fig[3, 2], "von Mises (MPa)", textsize = 20)
  # plot final topology
  heatmap(fig[4, 1], 1:FEAparams.meshSize[2], FEAparams.meshSize[1]:-1:1, top')
  # plot von mises field
  plotVM(FEAparams, disp, vf, fig, (4, 2))
  if goal == "save"
    # save image file
    save("C:/Users/LucasKaoid/Desktop/datasets/post/isolated features/imgs/$dataset $section $(string(sample)).png", fig)
  elseif goal == "display"
    display(fig)
  else
    println("\n\nWrong goal for plotSample().\n\n")
  end
end

# include plot of supports
function plotSupps(FEAparams, supps, fig)
  # Initialize support information variable
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
    [supports[supps[m, 1], supps[m, 2]] = 3 for m in 1:size(supps, 1)]
  end
  # plot supports
  heatmap(fig[2,1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,supports')
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
  Label(fig[1, 1], "Supports", textsize = 20)
  Label(fig[1, 2], "Force positions", textsize = 20)
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
  # labels for second line of grid
  Label(fig[3, 1], "Topology VF = $(round(vf;digits=3))", textsize = 20)
  # plot final topology
  heatmap(fig[4, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,top')
  # save image file
  save("C:/Users/LucasKaoid/Desktop/datasets/post/intermediateDensities/$(rand(1:99999)).png", fig)

end

# plot von Mises field
function plotVM(FEAparams, disp, vf, fig, figPos)
  # get von Mises field
  vm = calcVM(prod(FEAparams.meshSize), FEAparams, disp, 210e3*vf, 0.3)
  # plot von Mises
  _,hm = heatmap(fig[figPos[1], figPos[2]],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,vm')
  # setup colorbar for von Mises
  bigVal = 1.05*ceil(maximum(vm))
  t = floor(0.2*bigVal)
  t == 0 && (t = 1)
  Colorbar(fig[4, 3], hm, ticks = 0:t:bigVal)
end

# Make plot for VM model test
# Visually compare model predictions against truth
function plotVMtest(FEparams, vm, trueForces, model, modelName; folder = "")
  # Reshape output of stressCNN to match dataset format
  predForces = cpu(model(gpu(convert.(Float32, reshape(vm, (size(vm)..., 1, 1))))))
  # create makie figure and set it up
  fig = Figure(resolution = (1000, 700), fontsize = 20);
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
  Label(fig[3, 1], "Loss: "*sciNotation(Flux.mse(predForces, reshape(trueForces, (1, :))'), 3); align = (-1, 0.5), tellheight = :false)
  if length(folder) == 0
    Makie.save("./networks/trainingPlots/$modelName test.pdf", fig) # save figure created
  else
    mkpath(folder)
    Makie.save("$folder/$(rand(1:999999)).pdf", fig) # save figure created
  end
end

# Use a trained model to predict samples and make plots comparing
# with ground truth. In the end, combine plots into single pdf file
function stressCNNtestPlots(quant::Int, path::String, vm::Array{Float32, 4}, forceData::Array{Float32, 3}, finalName::String, FEparams, MLmodel)
  for sample in randDiffInt(quant, size(forceData, 3))
    plotVMtest(FEparams, vm[:, :, 1, sample], forceData[:, :, sample], MLmodel, 0; folder = path)
  end
  combinePDFs(path, finalName)
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