# Packages
println("Definitions...")
using Glob, Ferrite, LinearAlgebra, Makie, TopOpt, Graphs
using Parameters, Printf, HDF5, Statistics, Combinatorics
using Random, CUDA, Poppler_jll, MultivariateStats
using StatsBase, CairoMakie, MLUtils, Dates, Flux, GLMakie
using Traceur, ChainRulesCore, Zygote, ProfileView
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
using BSON: @load, @save
import Nonconvex
Nonconvex.@load NLopt
CairoMakie.activate!()
CUDA.allowscalar(false)
# function definitions in "utilities" folder
include.(glob("*", "./utilities/"));
# dataset path
const datasetPath = "C:/Users/LucasKaoid/Desktop/datasets/"
# reference number of channels used in TopologyGAN
const gf_dim::Int64 = 128
const df_dim::Int64 = 128
# coefficients used in TopologyGAN loss
const l1λ::Int32 = 10_000 
const l2λ::Int32 = 1
println("Done with definitions.")