# modules
using Makie, TopOpt, Parameters, StatProfilerHTML, Printf, HDF5, Statistics
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent;
import GLMakie, Nonconvex
Nonconvex.@load NLopt;
# function definitions
include("./QTOutils.jl")

if false
# struct with general parameters
@with_kw mutable struct FEAparameters
  quants::Int = 1 # number of TO problems per section
  V::Array{Real} = [0.4+rand()*0.5 for i in 1:quants] # volume fractions
  problems::Any = Array{Any}(undef, quants) # store FEA problem structs
  meshSize::Tuple{Int, Int} = (140, 50) # Size of rectangular mesh
  elementIDarray::Array{Int} = [i for i in 1:prod(meshSize)] # Vector that lists element IDs
  # matrix with element IDs in their respective position in the mesh
  elementIDmatrix::Array{Int,2} = convert.(Int, quad(meshSize...,[i for i in 1:prod(meshSize)]))
  section::Int = 1 # Number of dataset HDF5 files with "quants" samples each
end
FEAparams = FEAparameters()

# include toy problem in "problems" field to get mesh references
FEAparams = problem!(FEAparams)
end

# Function to be applied to every sample
function func(forces, supps, vf, disp, top, dataset, section, sample)

  

  return getNonBinaryTopos(forces, supps, vf, disp, top)

end

println(
  "call 'processDataset(func, id, numFiles)'.
  'func' - function to be applied to every sample
  'id' - name of folder
  'numFiles' - string with number of files to analyze in that folder, or 'end'"
)

@time [remSamples(f, "intermediateDensities/intermediateTopos") for f in 2:6]
combineFiles("intermediateDensities/intermediateTopos")
@time [processDataset(func, g, "end") for g in 1:6];

file = h5open("C:/Users/LucasKaoid/Desktop/datasets/data/analysis/37679", "r")
data = read.(HDF5.get_datasets(file))
close(file)

IDrem = findall(data[2] .> 0)
a = [data[h][IDrem] for h in keys(data)]

resultsFile = h5open("C:/Users/LucasKaoid/Desktop/datasets/post/intermediateDensities/asdf", "w")
create_dataset(resultsFile, "dataset", zeros(Int, length(IDrem)))
create_dataset(resultsFile, "section", zeros(Int, length(IDrem)))
create_dataset(resultsFile, "sampleID", zeros(Int, length(IDrem)))
create_dataset(resultsFile, "res", zeros(length(IDrem)))

for sample in keys(IDrem)
  resultsFile["dataset"][sample] = a[1][sample]
  resultsFile["res"][sample] = a[2][sample]
  resultsFile["section"][sample] = a[3][sample]
  resultsFile["sampleID"][sample] = a[4][sample]
end
close(resultsFile)

processDataset(func, 6, "end")