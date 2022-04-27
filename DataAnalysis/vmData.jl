using Glob, HDF5, StatsPlots, Statistics, DataFrames, IndexedTables

#=

  files = glob("*", "C:/Users/LucasKaoid/Desktop/datasets/data")[7:end] # get list of file names
  new = h5open("C:/Users/LucasKaoid/Desktop/datasets/data/yield", "w") # new file to store everything
  nSamples = 138600 # size of dataset
  # fields in new file
  create_dataset(new, "datasetSection", zeros(Int, nSamples))
  create_dataset(new, "sampleID", zeros(Int, nSamples))
  create_dataset(new, "vm", zeros(nSamples))
  # global sample counter
  count = 0
  for file in keys(files)
      @show file
      id = h5open(files[file], "r") # open current file
      # get data from current file
      ds = read(id["datasetSection"])
      sampleID = read(id["sampleID"])
      res = read(id["result"])
      # pass to new file
      new["datasetSection"][count+1:count+length(ds)] = ds
      new["sampleID"][count+1:count+length(ds)] = sampleID
      new["vm"][count+1:count+length(ds)] = res
      close(id) # close current file being read
      count += length(ds) # update global counter
  end
  close(new) # close and save new file

=#

new = h5open("C:/Users/LucasKaoid/Desktop/datasets/data/yield", "r") # open new file
# get data (new referring to the entire dataset)
ds = read(new["datasetSection"])
sampleID = read(new["sampleID"])
vm = read(new["vm"]) # (maximum von Mises of sample)/250
close(new) # close new file

showVal(x) = println(round.(x;digits=4)) # print function

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

