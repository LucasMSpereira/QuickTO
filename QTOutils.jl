# Packages
println("Definitions...")
using Glob, Ferrite, LinearAlgebra, Makie, TopOpt, Graphs, Zygote
using Parameters, Printf, HDF5, Statistics, Combinatorics, ChainRulesCore
using ProgressMeter, Random, CUDA, Poppler_jll, MultivariateStats
using StatsBase, CairoMakie, MLUtils, Dates, MetaGraphs, Flux
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
using BSON: @load, @save
import Nonconvex
Nonconvex.@load NLopt
using Base: llvmcall
CairoMakie.activate!()

# function definitions in "utilities" folder
include.(glob("*", "./utilities/"));

# dataset path
const datasetPath = "C:/Users/LucasKaoid/Desktop/datasets/"

println("Done with definitions.")