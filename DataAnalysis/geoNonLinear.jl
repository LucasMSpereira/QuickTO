include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")

function gNlinPlots(precision)

  file = h5open("C:/Users/LucasKaoid/Desktop/datasets/post/geomNonLinear/geomNonLinear", "r") # open current file
  # get data from file
  res = read(file["result"])
  close(file) # close current file
  quant = length(res) # quantity of samples
  ### Statistics
  print("mean: "); showVal(mean(res)) # 16.183
  print("std: "); showVal(std(res)) # 13.5279
  print("mean(res) + std(res): "); showVal(mean(res) + std(res)) # 29.711
  print("mean(res) - std(res): "); showVal(mean(res) - std(res)) # 2.655
  print("Maximum: "); showVal(maximum(res)) # 74.029
  print("Minimum: "); showVal(minimum(res)) # 3.014
  ### quantiles
  quantFig = Figure(resolution = (1400, 700), fontsize = 20)
  display(quantFig)
  ytik = 1:2:30
  ax1 = Axis(quantFig[1, 1], yticks = ytik, title = "Quantiles")
  ax2 = Axis(quantFig[1, 2], yticks = ytik, title = "Greatest values")
  ticks = 0:precision:1
  lines!(ax1, ticks, [quantile(res, ticks[i]) for i in keys(ticks)])
  n = 60  # greatest n values
  lines!(ax2, 1:n, sort(res; rev = true)[1:n])

  ####### quantiles
  println("\nQuantiles")
  precision = 0.1
  for i in 0:precision:1
    print(round(Int,i*100), "%\t")
    showVal(quantile(res, i))
  end
  
  ### Percentage of total dataset
  nSamples = 0 # count total amount of samples in dataset
  for folder in 1:6 # loop in datset folders
    # get list of file names in current folder
    files = glob("*", "C:/Users/LucasKaoid/Desktop/datasets/data/$folder")
    nSamples += numSample(files) # accumulate amount of samples across folder
  end
  ### Plots
  # fig = Figure(resolution = (1400, 700), fontsize = 20)
  # axBox = Axis(
  #   fig[1, 1], ylabel = "Norm of max nodal disp. per sample",
  #   title = "Boxplot",
  #   yticks = range(0, ceil(Int,maximum(res)); step = 10)
  # )
  # axHist = Axis(
  #   fig[1, 2], ylabel = "Fraction of samples",
  #   title = "Histogram",
  #   xticks = range(0, ceil(Int,maximum(res)); step = 10),
  #   yticks = range(0, 0.5; step = 0.05),
  #   xlabel = "Norm of max nodal disp. per sample",
  # )
  # b = boxplot!(fig[1, 1], ones(Int, quant), res; range = 1.5, gap = 0.1, color = :mediumseagreen)
  # h = hist!(fig[1, 2], res; bins = 10, normalization = :probability, color = :mediumseagreen)
  # save("geoNonLinear.png", quantFig)

end
@elapsed gNlinPlots(0.001)