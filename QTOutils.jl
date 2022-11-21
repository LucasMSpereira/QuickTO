# Packages
println("Packages...")
ENV["JULIA_CUDA_MEMORY_POOL"] = "none" # avoid GPU OOM issues
using Suppressor
@suppress_err begin
  using Glob, Ferrite, LinearAlgebra, Makie, TopOpt, Graphs
  using Parameters, Printf, HDF5, Statistics, Combinatorics
  using Random, CUDA, Poppler_jll, MultivariateStats
  using StatsBase, CairoMakie, MLUtils, Dates, Flux, GLMakie
  using Traceur, ChainRulesCore, Zygote, ProfileView
  using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
  using BSON: @load, @save
end
# import Nonconvex
# Nonconvex.@load NLopt
CairoMakie.activate!()
CUDA.allowscalar(false)
println("Utilities...")
# function and type definitions in "utilities" folder
readdir("./utilities"; join = true) |> x -> filter(y -> y[end - 2 : end] == ".jl", x) .|> include;
readdir("./utilities/ML utils"; join = true) .|> include;
readdir("./utilities/IO"; join = true) .|> include;
println("Constants...")
const datasetPath = "C:/Users/LucasKaoid/Desktop/datasets/" # dataset path
const datasetSize = 120_000 # approximate dataset size
# reference number of channels used in TopologyGAN
const gf_dim = 128
const df_dim = 128
println("Done with definitions.")