# Functions to generate/save plots

# plot forces
function plotForce(FEAparams, forces, fig)
  loadXcoord = zeros(size(forces, 1))
  loadYcoord = zeros(size(forces, 1))
  # Get loads positions
  for l in 1:size(forces, 1)
    loadXcoord[l] = forces[l,2]
    loadYcoord[l] = FEAparams.meshSize[2] - forces[l,1] + 1
  end
  # create and setup plotting axis
  axis = Axis(fig[2,2])
  xlims!(axis, -round(0.1*FEAparams.meshSize[1]), round(1.1*FEAparams.meshSize[1]))
  ylims!(axis, -round(0.1*FEAparams.meshSize[2]), round(1.1*FEAparams.meshSize[2]))
  # Plot loads as arrows
  arrows!(
    axis, loadXcoord, loadYcoord,
    forces[:,3], forces[:,4];
    # norm of weakest force. will be used to scale force vectors in arrows!() command
    lengthscale = 1 / (0.1*minimum( sqrt.( (forces[:,3]).^2 + (forces[:,4]).^2 ) ))
  )
  # text with values of force components
  f1 = "Forces (N):\n1: $(round(Int,forces[1, 3])); $(round(Int,forces[1, 4]))\n"
  f2 = "2: $(round(Int,forces[2, 3])); $(round(Int,forces[2, 4]))"
  l = text(
      fig[2,3],
      f1*f2;
  )
  textConfig(l)
  
end

# Line plots of evaluation histories
function plotLearnTries(trainParams, tries; drawLegend = true)
  f = Figure(resolution = (1000, 700));
  ax = Axis(f[1:2, 1], xlabel = "Validation epochs", ylabel = "Loss", title = timeNow())
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
      minimaText, "$i) Loss: $(valMin[1]) Epoch: $(valMin[2])"
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
  Makie.save("./networks/trainingPlots/$(timeNow()).pdf", f)
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
  plotForce(FEAparams, forces, fig)
  # labels for second line of grid
  Label(fig[3, 1], "Topology VF = $(round(vf; digits = 3))", textsize = 20)
  Label(fig[3, 2], "von Mises (MPa)", textsize = 20)
  # plot final topology
  heatmap(fig[4, 1], 1:FEAparams.meshSize[2], FEAparams.meshSize[1]:-1:1, top')
  # plot von mises field
  plotVM(FEAparams, disp, vf, fig)
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
function plotVM(FEAparams, disp, vf, fig)
  # get von Mises field
  vm = calcVM(prod(FEAparams.meshSize), FEAparams, disp, 210e3*vf, 0.3)
  # plot von Mises
  _,hm = heatmap(fig[4, 2],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,vm')
  # setup colorbar for von Mises
  bigVal = 1.05*ceil(maximum(vm))
  t = floor(0.2*bigVal)
  t == 0 && (t = 1)
  Colorbar(fig[4, 3], hm, ticks = 0:t:bigVal)
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