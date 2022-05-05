# Functions to generate/save plots

# create figure to vizualize sample
function plotSample(FEAparams)
  # Vector of strings with paths to each dataset file
  files = glob("*", "C:/Users/LucasKaoid/Desktop/datasets/data")
  lines = FEAparams.meshSize[2]
  quantForces = 2
  colSize = 500
  count = 0
  for file in keys(files)
  # open file and read data to be plotted
    forces, supps, vf, disp, top = getDataFSVDT(files[file])
    file == 1 && (FEAparams.problems[1] = rebuildProblem(vf[1], supps[:,:,1], forces[:,:,1]))
    dataset, section = getIDs(files[file])
    mkpath("C:/Users/LucasKaoid/Desktop/datasets/fotos/$dataset")
    for i in 1:size(top,3)
      # only generate images for a fraction of samples
      rand() > 0.01 && continue
      count += 1
      vm = calcVM(prod(FEAparams.meshSize), FEAparams, disp[:,:,(2*i-1):(2*i)], 210e3*vf[i], 0.3)
      println("image $count            $( round(Int, file/length(files)*100) )%")
      # create makie figure and set it up
      fig = Figure(resolution = (1400, 700));
      colsize!(fig.layout, 1, Fixed(colSize))
      # labels for first line of grid
      Label(fig[1, 1], "Supports", textsize = 20)
      Label(fig[1, 2], "Force positions", textsize = 20)
      colsize!(fig.layout, 2, Fixed(colSize))
      supports = zeros(FEAparams.meshSize)'
      if supps[:,:,i][1,3] > 3


        if supps[:,:,i][1,3] == 4
          # left
          supports[:,1] .= 3
        elseif supps[:,:,i][1,3] == 5
          # bottom
          supports[end,:] .= 3
        elseif supps[:,:,i][1,3] == 6
          # right
          supports[:,end] .= 3
        elseif supps[:,:,i][1,3] == 7
          # top
          supports[1,:] .= 3
        end


      else

        [supports[supps[:,:,i][m, 1], supps[:,:,i][m, 2]] = 3 for m in 1:size(supps[:,:,i])[1]]

      end
      # plot supports
      heatmap(fig[2,1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,supports')
      # plot forces
      loadXcoord = zeros(quantForces)
      loadYcoord = zeros(quantForces)
      for l in 1:quantForces
        loadXcoord[l] = forces[:,:,i][l,2]
        loadYcoord[l] = lines - forces[:,:,i][l,1] + 1
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
      # text with values of force components
      f1 = "Forces (N):\n1: $(round(Int,forces[:,:,i][1, 3])); $(round(Int,forces[:,:,i][1, 4]))\n"
      f2 = "2: $(round(Int,forces[:,:,i][2, 3])); $(round(Int,forces[:,:,i][2, 4]))"
      l = text(
          fig[2,3],
          f1*f2;
      )

      #
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
      #
      
      # labels for second line of grid
      Label(fig[3, 1], "Topology VF = $(round(vf[i];digits=3))", textsize = 20)
      Label(fig[3, 2], "von Mises (MPa)", textsize = 20)
      # plot final topology
      heatmap(fig[4, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,top[:,:,i]')
      # plot von Mises
      _,hm = heatmap(fig[4, 2],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,vm')
      # setup colorbar for von Mises
      bigVal = ceil(maximum(vm))
      t = floor(0.2*bigVal)
      t == 0 && (t = 1)
      Colorbar(fig[4, 3], hm, ticks = 0:t:bigVal)
      # save image file
      save("C:/Users/LucasKaoid/Desktop/datasets/fotos/$dataset/$(section * " " * string(i)).png", fig)
    end
  end
  println()
end

# test bounds for detecting topologies with too many intermediate densities
function plotTopoIntermediate(forces, supps, vf, top, FEAparams, bound)
  
  lines = FEAparams.meshSize[2]
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
    loadYcoord[l] = lines - forces[l,1] + 1
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

  #
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
  #
  
  # labels for second line of grid
  Label(fig[3, 1], "Topology VF = $(round(vf;digits=3))", textsize = 20)
  # plot final topology
  heatmap(fig[4, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,top')
  # save image file
  save("C:/Users/LucasKaoid/Desktop/datasets/post/intermediateDensities/$(rand(1:99999)).png", fig)

end