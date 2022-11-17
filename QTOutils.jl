# Packages
println("Packages...")
ENV["JULIA_CUDA_MEMORY_POOL"] = "none"
# ENV["CUARRAYS_MEMORY_POOL"] = split
using Glob, Ferrite, LinearAlgebra, Makie, TopOpt, Graphs
using Parameters, Printf, HDF5, Statistics, Combinatorics
using Random, CUDA, Poppler_jll, MultivariateStats
using StatsBase, CairoMakie, MLUtils, Dates, Flux, GLMakie
using Traceur, ChainRulesCore, Zygote, ProfileView
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
using BSON: @load, @save
# import Nonconvex
# Nonconvex.@load NLopt
CairoMakie.activate!()
CUDA.allowscalar(false)
println("Utilities...")
# function and type definitions in "utilities" folder
include("./utilities/feaFuncs.jl");
readdir("./utilities/ML utils"; join = true) .|> include;
println("Constants...")
# dataset path
const datasetPath = "C:/Users/LucasKaoid/Desktop/datasets/"
# reference number of channels used in TopologyGAN
const gf_dim = 128
const df_dim = 128
# coefficients used in TopologyGAN loss
const l1λ = 10_000 
const l2λ = 1
println("Done with definitions.")