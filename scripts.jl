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
  return topoFEA(forces, supps, vf, top)
end

println(
  "call 'processDataset(func, id, numFiles)'.
  'func' - function to be applied to every sample
  'id' - name of folder
  'numFiles' - string with number of files to analyze in that folder, or 'end'
    to analyze all"
)

file = h5open("C:/Users/LucasKaoid/Desktop/datasets/post/geomNonLinear/geomNonLinear", "r")
ds = read(file["dataset"])
res = read(file["result"])
sID = read(file["sampleID"])
sec = read(file["section"])
close(file)

pset = PointSet(rand(Point2, 100))
@time chul = hull(pset, GrahamScan())

fig = GLMakie.Figure(resolution = (800, 400))
viz(fig[1,1], chul)
viz!(fig[1,1], pset, color = :black)