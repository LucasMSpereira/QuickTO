using Glob, HDF5, StatsPlots, Statistics, DataFrames, IndexedTables
include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")

new = h5open("C:/Users/LucasKaoid/Desktop/datasets/data/yield", "r") # open new file
# get data (new referring to the entire dataset)
ds = read(new["datasetSection"])
sampleID = read(new["sampleID"])
vm = read(new["vm"]) # (maximum von Mises of sample)/250
close(new) # close new file

##### Statistics
@show mean(vm) # 0.35267
@show std(vm) # 0.16245
@show mean(vm) + std(vm) # 0.515
@show mean(vm) - std(vm) # 0.19
vmMax = maximum(vm)
println("Maximum vm/250: $(round(vmMax;digits=3))")  # 9.851
vmMin = minimum(vm)
println("Minimum vm/250: $(round(vmMin;digits=3))")  # 0.01968

##### plots
vmNoMax = filter(x->x!=vmMax,vm)
vmElastic = filter(x->x<1,vm)
scalefontsizes()
scalefontsizes(1.3)
vmTabNoMax = DataFrame(vonMises=vmNoMax)
vmTabElastic = DataFrame(vonMises=vmElastic)
vmTab = DataFrame(vonMises=vm)

function myPlots(data, dataTab)
h = histogram(
  data; bins=range(0, stop = maximum(data), length = 10),
  title = "Max. von Mises/250 histogram",
  ylabel = "Fraction of samples",
  normalize = :probability,
  legend = false
)
b = @df dataTab boxplot(
  :vonMises;
  title = "Boxplot",
  legend = false,
  outliers = false,
  # whisker_range = 4
)
p = plot(
  h,b;
  size=(900,900),
  tickfontsize = 14
)
display(p)
savefig("vm.png")
end
myPlots(vmElastic, vmTabElastic)

####### quantiles
println("\nQuantiles")
precision = 0.1
for i in 0:precision:1
  print(round(Int,i*100), "%\t")
  showVal(quantile(vm, i))
end

# percentage of plastic cases
print("Percentage of plastic cases: ")
println(round(length(filter(x -> x>1, vm)) / length(vm) * 100;digits=3), "%") # 0.686%

