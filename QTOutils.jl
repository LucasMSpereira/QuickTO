# Packages
println("Definitions...")
using Glob, Ferrite, LinearAlgebra, Makie, TopOpt, Graphs, MetaGraphs, Flux
using Parameters, Printf, HDF5, Statistics, Combinatorics, MultivariateStats
using ProgressMeter, Random, TensorBoardLogger, MLDatasets, BSON, CUDA
using StatsBase, CairoMakie, MLUtils, Hyperopt, Dates, UnPack
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
import Nonconvex
Nonconvex.@load NLopt
CairoMakie.activate!()

# function definitions in "utilities" folder
include.(glob("*", "./utilities/"));
println("Done with definitions.")