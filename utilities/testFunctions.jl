# Temporary functions used for testing hypotheses and debugging

function displayDispPlotTest(FEparams, disp, trueForces, model, lossFun)
  # Reshape output of loadCNN to match dataset format
  predForces = convert.(Float32, reshape(disp, (size(disp)..., 1))) |> gpu |> model |> cpu
  fig = Figure(resolution = (1000, 700), fontsize = 20) # create makie figure and set it up
  textPos = (-0.1, 0.5) # text position in final figure
  forceAxis = plotForce(FEparams, reduce(hcat, target), fig, (1, 2:3), (2, 2); alignText = textPos, axisHeight = 200) # plot true forces
  # plot predicted forces
  plotForce(FEparams, reshapeForces(predForces), fig, (1, 2:3), (2, 3); newAxis = forceAxis, paintArrow = :green, paintText = :green, alignText = textPos)
  # Labels
  Label(fig[1, 1], "Force positions"; tellheight = :false)
  (colsize!(fig.layout, i, Fixed(500)) for i in 1:2)
  # loss value of current prediction
  @show predForces; @show reshape(trueForces, (1, :))'
  @show lossFun(predForces, reshape(trueForces, (1, :))')
  Label(fig[2, 1], "Loss: "*sciNotation(lossFun(predForces, reshape(trueForces, (1, :))'), 3);
  align = (-1, 0.5), tellheight = :false)
  display(fig)
end

function loadPosTest(nels)
  # Random ID(s) to choose element(s) to be loaded
  global loadElements = rand(1:nels,2)
  # Matrices to indicate position and component of load
  global Fx = zeros((140,50))'
  global Fy = zeros((140,50))'
  # i,j mesh positions of chosen elements
  global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
  @show loadPoss
  # Generate point load component values
  randLoads = (-ones(length(loadElements),2) + 2*rand(length(loadElements),2))*1e3/6
  @show round.(Int,randLoads)
  # Insert load components in respective matrices
  [Fx[loadPoss[i]] = randLoads[i,1] for i in keys(loadPoss)];
  [Fy[loadPoss[i]] = randLoads[i,2] for i in keys(loadPoss)];
  # Get vector with IDs of loaded nodes
  grid = generate_grid(Quadrilateral, (140,50))
  myCells = [grid.cells[g].nodes for g in loadElements]
  @show myCells
  pos = reshape([myCells[ele][eleNode] for eleNode in 1:4, ele in keys(loadElements)], (:,1))
  @show pos
  return pos, Fx, Fy, randLoads
end

function miniPlot(vals, folderName, i, j, vf)
  fig = Figure(resolution = (900, 600));
  # display(fig)
  bigVal = round(maximum(vals))
  colSize = 500
  l = text(
        fig[1,1],
        "max: $(round(bigVal))\navg: $(round(mean(vals)))\nstd: $(round(std(vals)))\nVF: $(round(vf; digits=2))";
  )
  colsize!(fig.layout, 1, Fixed(150))

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

  _,hm = heatmap(fig[1,2],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,vals')
  colsize!(fig.layout, 2, Fixed(colSize))
  t = round(0.2*bigVal)
  t == 0 && (t = 1)
  Colorbar(fig[1, 3], hm, ticks = 0:t:bigVal)
  save("C:/Users/LucasKaoid/Desktop/datasets/$(folderName)/fotos/sample $i/$j.png", fig)
end

function plotCheck(forces, vals, sample; quants=3)
  lines = size(vals,1)
  cols = size(vals,2)
  numForces = size(forces,1)
  # check sample
  result, sProds, grads, avgs, vmRatio, loadRatio, ratioRatio = checkSample(numForces, vals, sample, quants, forces)
  # figure creation and setup
  fig = Figure(resolution = (1500, 900));
  display(fig)
  colSize = 500
  colsize!(fig.layout, 1, 600)
  # include text with dot product values
  l = text(
    fig[1,3],
    # "Grads :\n1: $(round(Int,grads[1,1])); $(round(Int,grads[2,1]))\n2: $(round(Int,grads[1,2])); $(round(Int,grads[2,2]))\nForces :\n1: $(round(Int,forces[1,3])); $(round(Int,forces[1,4]))\n2: $(round(Int,forces[2,3])); $(round(Int,forces[2,4]))";
    "dot1: $(round(abs(sProds[1]);digits=3))\ndot2: $(round(abs(sProds[2]);digits=3))";
    # position = Point2f0(50.0, 50.0)
  )
  colsize!(fig.layout, 3, 100)

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
    # l = Label(fig[2,3], "JOOOOOOOJ\nJOOOOOJOJOJOJ"; tellheight = false, tellwidth = false, halign = :left)
  #

  # heatmap of physical scalar field (usually von Mises stress)
  heatmap(fig[1,1],1:lines,cols:-1:1,vals[:,:,sample]')
  # axis used to draw vector of physical field gradients
  axis = Axis(fig[1,2])
  # choose background color based on result of sample check
  result ? (axis.attributes.backgroundcolor = :seagreen) : (axis.attributes.backgroundcolor = :tomato2)
  xlims!(axis, -round(Int, 0.1*cols), round(Int, 1.1*cols))
  ylims!(axis, -round(Int, 0.1*lines), round(Int, 1.1*lines))
  colsize!(fig.layout, 2, Fixed(colSize))
  # build vectors with load positions
  loadXcoord = zeros(numForces)
  loadYcoord = zeros(numForces)
  for l in 1:numForces
    loadXcoord[l] = forces[:,:,sample][l,2]
    loadYcoord[l] = lines - forces[:,:,sample][l,1] + 1
  end
  # plot gradient vectors
  arrows!(
    axis, loadXcoord, loadYcoord,
    grads[1,:],  grads[2,:];
    lengthscale = 3
  )
  # norm of weakest force. will be used to scale force vectors in arrows!() command
  fmin = 0.2*minimum(sqrt.((forces[:,:,sample][:,3]).^2+(forces[:,:,sample][:,4]).^2))
  # axis used to draw load vectors
  axis2 = Axis(fig[2,2])
  xlims!(axis2, -round(Int, 0.1*cols), round(Int, 1.1*cols))
  ylims!(axis2, -round(Int, 0.1*lines), round(Int, 1.1*lines))
  # plot vector of loads
  arrows!(
    axis2, loadXcoord, loadYcoord,
    forces[:,:,sample][:,3], forces[:,:,sample][:,4];
    lengthscale = 1/fmin
  )
  # include text with force components
  vmAvgs = "vm avg 1: $(round(avgs[1];digits=2))\nvm avg 2: $(round(avgs[2];digits=2))\n"
  vmRatioText = "ratio of stress avgs: $(round(vmRatio;digits=2))\n"
  loadRatioText = "ratio of load norms: $(round(loadRatio;digits=2))\n"
  ratioRatioText = "ratio of ratios: $(round(ratioRatio;digits=2))"
  l = text(
        fig[2,1],
        prod([vmAvgs vmRatioText loadRatioText ratioRatioText]);
        # position = Point2f0(50.0, 50.0)
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
    l.axis.attributes.halign = :left
    l.axis.attributes.width = 600
    # l = Label(fig[2,3], "JOOOOOOOJ\nJOOOOOJOJOJOJ"; tellheight = false, tellwidth = false, halign = :left)
  #

end

function plotSampleTest(sample, folderName, FEAparams)
  # open file and read data to be plotted
  id = h5open("C:/Users/LucasKaoid/Desktop/datasets/$(folderName)/$(folderName)data", "r")
  global top = read(id["topologies"])
  global forces = read(id["inputs"]["forces"])
  global supps = read(id["inputs"]["dispBoundConds"])
  global vf = read(id["inputs"]["VF"])
  global vm = read(id["conditions"]["vonMises"])
  close(id)
  # create makie figure and set it up
  print("Generating image of samples: ")
  lines = size(top)[1]
  quantForces = size(forces)[1]
  colSize = 250
  for i in 1:sample
    # only generate images for a fraction of samples
    # rand() > 0.1 && continue
    print("$i   ")
    fig = Figure(resolution = (1400, 700));
    colsize!(fig.layout, 1, Fixed(colSize))
    # display(fig)
    # labels for first line of grid
    Label(fig[1, 1], "supports", textsize = 20)
    Label(fig[1, 2], "force positions", textsize = 20)
    colsize!(fig.layout, 2, Fixed(colSize))
    # plot support(s) and force locations
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
    heatmap(fig[2, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,supports')
    global loadXcoord = zeros(quantForces)
    global loadYcoord = zeros(quantForces)
    for l in 1:quantForces
      global loadXcoord[l] = forces[:,:,i][l,2]
      global loadYcoord[l] = lines - forces[:,:,i][l,1] + 1
    end
    # norm of weakest force. will be used to scale force vectors in arrows!() command
    fmin = 0.2*minimum(sqrt.((forces[:,:,i][:,3]).^2+(forces[:,:,i][:,4]).^2))
    axis = Axis(fig[2,2])
    xlims!(axis, -round(0.1*FEAparams.meshSize[1]), round(1.1*FEAparams.meshSize[1]))
    ylims!(axis, -round(0.1*FEAparams.meshSize[2]), round(1.1*FEAparams.meshSize[2]))
    arrows!(
      axis, loadXcoord, loadYcoord,
      forces[:,:,i][:,3], forces[:,:,i][:,4];
      lengthscale = 1/fmin
    )
    # labels for second line of grid
    Label(fig[3, 1], "topology VF = $(round(vf[i];digits=3))", textsize = 20)
    Label(fig[3, 2], "von Mises (MPa)", textsize = 20)
    # plot final topology and von Mises, indicating final volume fraction
    heatmap(fig[4, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,top[:,:,i]')
    _,hm = heatmap(fig[4, 2],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,vm[:,:,i]')
    bigVal = round(maximum(vm[:,:,i]))
    t = round(0.2*bigVal)
    t == 0 && (t = 1)
    Colorbar(fig[4, 3], hm, ticks = 0:t:bigVal)
    colsize!(fig.layout, 2, Fixed(colSize))
    # text with values of force components
    l = text(
        fig[2,3],
        "Forces (N):\n1: $(round(Int,forces[:,:,i][1, 3])); $(round(Int,forces[:,:,i][1, 4]))\n2: $(round(Int,forces[:,:,i][2, 3])); $(round(Int,forces[:,:,i][2, 4]))";
        # position = Point2f0(50.0, 50.0)
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
        # l = Label(fig[2,3], "JOOOOOOOJ\nJOOOOOJOJOJOJ"; tellheight = false, tellwidth = false, halign = :left)
    #
    
    save("C:/Users/LucasKaoid/Desktop/datasets/$(folderName)/fotos/sample $i.png", fig)
  end
  println()
end

# print force matrices for testing
function printForces(target, out)
  t = reduce(hcat, target)
  println("target")
  [println(t[l, :]) for l in 1:2]
  println("output")
  o = reduce(hcat, out)
  [println(o[l, :]) for l in 1:2]
end

# print loss calculation intermediate values
function printLoss(lossFun, out, target)
  print()
  lossFun(out, target)
  lossFun(out, target)
  lossFun(out, target)
  lossFun(out, target)
  mean([ ])
end

function testSamplePlot(folderName, FEAparams, i)
  # open file and read data to be plotted
  id = h5open("C:/Users/LucasKaoid/Desktop/datasets/$(folderName)/$(folderName)data2", "r")
  global top = read(id["topologies"])
  global forces = read(id["inputs"]["forces"])
  global supps = read(id["inputs"]["dispBoundConds"])
  global vf = read(id["inputs"]["VF"])
  global vm = read(id["conditions"]["vonMises"])
  close(id)
  lines = size(top)[1]
  quantForces = size(forces)[1]
  colSize = 250
    fig = Figure(resolution = (900, 700));
    colsize!(fig.layout, 1, Fixed(colSize))
    display(fig)
    # labels for first line of grid
    Label(fig[1, 1], "supports", textsize = 20)
    Label(fig[1, 2], "force positions", textsize = 20)
    colsize!(fig.layout, 2, Fixed(colSize))
    # plot support(s) and force locations
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
    heatmap(fig[2, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,supports')
    global loadXcoord = zeros(quantForces)
    global loadYcoord = zeros(quantForces)
    for l in 1:quantForces
      global loadXcoord[l] = forces[:,:,i][l,2]
      global loadYcoord[l] = lines - forces[:,:,i][l,1] + 1
    end
    # norm of weakest force. will be used to scale force vectors in arrows!() command
    fmin = 0.6*minimum(sqrt.((forces[:,:,i][:,3]).^2+(forces[:,:,i][:,4]).^2))
    axis = Axis(fig[2,2])
    xlims!(axis, -round(0.1*FEAparams.meshSize[1]), round(1.1*FEAparams.meshSize[1]))
    ylims!(axis, -round(0.1*FEAparams.meshSize[2]), round(1.1*FEAparams.meshSize[2]))
    arrows!(
      axis, loadXcoord, loadYcoord,
      forces[:,:,i][:,3], forces[:,:,i][:,4];
      lengthscale = 1/fmin
    )
    # labels for second line of grid
    Label(fig[3, 1], "topology VF = $(round(vf[i];digits=3))", textsize = 20)
    Label(fig[3, 2], "von Mises (MPa)", textsize = 20)
    # plot final topology and von Mises, indicating final volume fraction
    heatmap(fig[4, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,top[:,:,i]')
    _,hm = heatmap(fig[4, 2],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,vm[:,:,i]')
    bigVal = round(maximum(vm[:,:,i]))
    t = round(0.2*bigVal)
    t == 0 && (t = 1)
    Colorbar(fig[4, 3], hm, ticks = 0:t:bigVal)
    colsize!(fig.layout, 2, Fixed(colSize))
    # text with values of force components
    l = text(
        fig[2,3],
        "Forces (N):\n1: $(round(Int,forces[:,:,i][1, 3])); $(round(Int,forces[:,:,i][1, 4]))\n2: $(round(Int,forces[:,:,i][2, 3])); $(round(Int,forces[:,:,i][2, 4]))";
        # position = Point2f0(50.0, 50.0)
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
        # l = Label(fig[2,3], "JOOOOOOOJ\nJOOOOOJOJOJOJ"; tellheight = false, tellwidth = false, halign = :left)
    #
    
end

function valPlot(vals, FEAparams)
  fig = Figure(resolution = (800, 600));
  display(fig);
  bigVal = round(maximum(vals));
  _,hm = heatmap(fig[1,1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,vals');
  colsize!(fig.layout, 1, Fixed(500));
  t = round(0.2*bigVal);
  t == 0 && (t = 1);
  Colorbar(fig[1, 2], hm, ticks = 0:t:bigVal);
end

function writeStressesTest(nels, FEAparams, disp, problemID, e, v)
  # "Programming the finite element method", 5. ed, Wiley, pg 35
  state = "stress"
  principals = Array{Real}(undef, nels, 2) # principal stresses
  σ = Array{Real}(undef, nels) # stresses
  strainEnergy = zeros(FEAparams.meshSize)' # strain energy density in each element
  vm = zeros(FEAparams.meshSize)' # von Mises for each element
  centerDispGrad = Array{Real}(undef, nels, 2)
  cellValue = CellVectorValues(QuadratureRule{2, RefCube}(2), Lagrange{2,RefCube,1}())
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

  return vm, quad(FEAparams.meshSize...,σ), quad(FEAparams.meshSize...,principals[:, 1]), quad(FEAparams.meshSize...,principals[:, 2]), strainEnergy

end