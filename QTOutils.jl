# Packages
println("Definitions...")
using Glob, Ferrite, LinearAlgebra, Makie, TopOpt, Graphs
using Parameters, Printf, HDF5, Statistics, Combinatorics
using ProgressMeter, Random, CUDA, Poppler_jll, MultivariateStats
using StatsBase, CairoMakie, MLUtils, Dates, MetaGraphs, Flux
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
using BSON: @load, @save
import Nonconvex
Nonconvex.@load NLopt
CairoMakie.activate!()

# function definitions in "utilities" folder
include.(glob("*", "./utilities/"));
println("Done with definitions.")