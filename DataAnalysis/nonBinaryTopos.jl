using Glob, GLMakie, Statistics
include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")

function nbtPlots()

  file = h5open("C:/Users/LucasKaoid/Desktop/datasets/post/intermediateDensities/intermediateTopos", "r") # open current file
  # get data from file
  ds = read(file["dataset"])
  ss = read(file["section"])
  sample = read(file["sampleID"])
  res = read(file["intermPercent"])
  close(file) # close current file
  quant = length(ds) # quantity of samples
  ### Statistics
  print("mean: ")
  showVal(mean(res)) # 16.183
  print("std: ")
  showVal(std(res)) # 13.5279
  print("mean(res) + std(res): ")
  showVal(mean(res) + std(res)) # 29.711
  print("mean(res) - std(res): ")
  showVal(mean(res) - std(res)) # 2.655
  print("Maximum: ")
  showVal(maximum(res)) # 74.029
  print("Minimum: ")
  showVal(minimum(res)) # 3.014
  ### quantiles
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
  print("\nPercentage of dataset: $(round(quant/nSamples*100;digits=1))%")
  ### Plots
  fig = Figure(resolution = (1400, 700), fontsize = 20)
  axBox = Axis(
    fig[1, 1], ylabel = "Percentage of intermediate densities per sample",
    title = "Boxplot",
    yticks = range(0, ceil(Int,maximum(res)); step = 10)
  )
  axHist = Axis(
    fig[1, 2], ylabel = "Fraction of samples",
    title = "Percentage of elements with intermediate densities",
    xticks = range(0, ceil(Int,maximum(res)); step = 10),
    yticks = range(0, 0.5; step = 0.05),
    xlabel = "Percentage of intermediate densities per sample",
  )
  b = boxplot!(fig[1, 1], ones(Int, quant), res; range = 1.5, gap = 0.1, color = :mediumseagreen)
  h = hist!(fig[1, 2], res; bins = 10, normalization = :probability, color = :mediumseagreen)
  display(fig)

  return fig, b, h, axBox

end
@elapsed obj = nbtPlots()