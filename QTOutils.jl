# Packages
using Suppressor
@time begin
  println("Packages...")
  ENV["JULIA_CUDA_MEMORY_POOL"] = "none" # avoid GPU OOM issues
  Suppressor.@suppress_err begin
    using LinearAlgebra, Makie, TopOpt, Ferrite, JLD2, TimerOutputs
    using Parameters, HDF5, Statistics, BSON, ValueHistories, ShapML
    using CUDA, Poppler_jll, MultivariateStats, Random, Colors
    using StatsBase, CairoMakie, MLUtils, Dates, Flux, GLMakie
    using Zygote, Optimisers, ChainRulesCore, Interpolations, DataFrames
    using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
  end
end
# import Nonconvex
# Nonconvex.@load NLopt
CUDA.allowscalar(false)
println("Utilities...")
# function and type definitions in "utilities" folder
readdir("./utilities"; join = true) |> x -> filter(y -> y[end - 2 : end] == ".jl", x) .|> include;
readdir("./utilities/ML utils"; join = true) .|> include;
readdir("./utilities/IO"; join = true) .|> include;
println("Constants...")
# reference number of channels used in TopologyGAN
const gf_dim = 128
const df_dim = 128
const datasetNonTestSize = 106_336 # number of samples for training and validation
# series used for interpolation of physical fields
const centroidY = 0.5:1:49.5
const centroidX = 0.5:1:139.5
const nodeY = 0:50
const nodeX = 0:140
println("Done with definitions.")