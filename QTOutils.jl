using Glob, Ferrite, LinearAlgebra, Makie, TopOpt, Graphs, MetaGraphs
using Parameters, Printf, HDF5, Statistics, GLMakie, Combinatorics
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
import Nonconvex
Nonconvex.@load NLopt;

# function definitions
include("./utilities/feaFuncs.jl")
include("./utilities/funcs.jl")
include("./utilities/io.jl")
include("./utilities/postProcess.jl")
include("./utilities/testFunctions.jl")
include("./utilities/plotFuncs.jl")