# Install packages
import Pkg
println("\nPACKAGES...\n")
if !packagesInstalled
  map(
    Pkg.add,
    (
      "TopOpt", "Makie", "Ferrite", "JLD2", "TimerOutputs",
      "Parameters", "HDF5", "Statistics", "BSON", "ValueHistories", "ShapML",
      "Poppler_jll", "MultivariateStats", "Colors", "Suppressor",
      "CairoMakie", "MLUtils", "Flux", "GLMakie", "ExplainableAI",
      "Zygote", "Optimisers", "ChainRulesCore", "Interpolations", "DataFrames"
    )
  )
end
ENV["JULIA_CUDA_MEMORY_POOL"] = "none" # avoid GPU OOM issues
# Use packages
using Suppressor
@suppress_err begin
  using LinearAlgebra, Dates, TopOpt, Makie, Ferrite, JLD2, TimerOutputs
  using Parameters, HDF5, Statistics, BSON, ValueHistories, ShapML
  using CUDA, Poppler_jll, MultivariateStats, Random, Colors
  using StatsBase, CairoMakie, MLUtils, Flux, GLMakie, ExplainableAI
  using Zygote, Optimisers, ChainRulesCore, Interpolations, DataFrames
  using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
end
CUDA.allowscalar(false)
println("\nDEFINITIONS...\n")
# function and type definitions in "utilities" folder
utilsPath = readdir("./QuickTO/utilities"; join = true) |> x -> filter(y -> y[end - 2 : end] == ".jl", x)
utilsPath .|> x -> "." * x[10 : end] .|> include
MLutilsPath = readdir("./QuickTO/utilities/ML utils"; join = true)
MLutilsPath .|> x -> "." * x[10 : end] .|> include
IOutilsPath = readdir("./QuickTO/utilities/IO"; join = true)
IOutilsPath .|> x -> "." * x[10 : end] .|> include
println("\nCONSTANTS...\n")
const datasetPath = "C:/Users/LucasKaoid/Desktop/datasets/" # dataset path
const datasetNonTestSize = 106_336 # number of samples for training and validation
# reference number of channels used in TopologyGAN
const gf_dim = 128
const df_dim = 128
# series used for interpolation of physical fields
const centroidY = 0.5:1:49.5
const centroidX = 0.5:1:139.5
const nodeY = 0:50
const nodeX = 0:140
println("DONE WITH DEFINITIONS.")