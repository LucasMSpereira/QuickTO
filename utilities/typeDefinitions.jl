# Custom type definitions

abstract type trainConfig end

# Configurations for training of neural networks for fixed number of epochs
mutable struct epochTrainConfig <: trainConfig
  const epochs::Int64 # Total number of training epochs
  const validFreq::Int64 # Evaluation frequency in epochs
  const schedule::Int64 # Learning rate adjustment interval in epochs. 0 for no scheduling
  const decay::Float64 # If scheduling, periodically multiply learning rate by this value
  evaluations::Array{<:Real} # History of evaluation losses
end
epochTrainConfig(
  epochs, validFreq; schedule = 0, decay = 0.0, evaluations = Float64[],
) = epochTrainConfig(epochs, validFreq, schedule, decay, evaluations)

# Configurations for early-stop training of neural networks
mutable struct earlyStopTrainConfig <: trainConfig
  const validFreq::Int64 # Evaluation frequency in epochs
  const decay::Float64 # If scheduling, periodically multiply learning rate by this value
  const schedule::Int32 # Learning rate adjustment interval in epochs. 0 for no scheduling
  evaluations::Array{<:Real} # History of evaluation losses
  # Interval of most recent validations used for early stop criterion
  const earlyStopQuant::Int32
  # Minimal percentage drop in loss in the last "earlyStopQuant" validations
  const earlyStopPercent::Float32
end

earlyStopTrainConfig(
  validFreq; decay = 0.0, schedule = 0, evaluations = Float64[],
  earlyStopQuant = 3, earlyStopPercent = 1
) = earlyStopTrainConfig(validFreq, decay, schedule, evaluations,
  earlyStopQuant, earlyStopPercent
)

mutable struct optimisationInfo
  opt::Flux.Optimise.AbstractOptimiser # optimizer used in training
  optState::NamedTuple # optimizer's state
end

mutable struct nnInfo
  neuralNetwork::Chain # network
  optInfo::optimisationInfo # optimisation setup
  nnValues # log values during training
end

# Meta-data for GAN pipeline
mutable struct GANmetaData
  genDefinition::nnInfo # generator information
  discDefinition::nnInfo # discriminator information
  const trainConfig::trainConfig # parameters for training
  lossesVals::Dict{Symbol, Vector{Float64}} # loss histories
  files::Dict{Symbol, Vector{String}}
  const datasetUsed::Float64 # fraction of dataset used
  # number of critic iters. between gen iters. when using wasserstein loss
  nCritic::Int32
end

## GANmetaData APIs
# save histories of losses
function (ganMD::GANmetaData)(valLossHist::NTuple{2, Float32}; context = :validate)
  if context == :validate
    push!(ganMD.lossesVals[:genValHistory], valLossHist[1])
    push!(ganMD.lossesVals[:discValHistory], valLossHist[2])
  elseif context == :test
    push!(ganMD.lossesVals[:genTest], valLossHist[1])
    push!(ganMD.lossesVals[:discTest], valLossHist[2])
  else
    println("Wrong 'context' input for saving loss history (GANmetaData type API).")
  end
  return nothing
end

# Outer constructor to create object in the begining.
# Used when training new models
GANmetaData(
  generator::Chain, discriminator::Chain,
  genOpt::Flux.Optimise.AbstractOptimiser, discOpt::Flux.Optimise.AbstractOptimiser,
  myTrainConfig::trainConfig
) = GANmetaData(
  nnInfo(generator, optimisationInfo(genOpt, Optimisers.setup(genOpt, generator)), MVHistory()),
  nnInfo(discriminator, optimisationInfo(discOpt, Optimisers.setup(discOpt, discriminator)), MVHistory()),
  myTrainConfig,
  Dict(
    :genValHistory => Float64[],
    :discValHistory => Float64[],
    :genTest => Float64[],
    :discTest => Float64[]
  ),
  if runningInColab == false # if running locally
    getNonTestFileLists(datasetPath * "data/trainValidate", 0.7)
  else # if running in colab
    getNonTestFileLists("./gdrive/MyDrive/dataset files/trainValidate", 0.7)
  end,
  percentageDataset, 5
)

# Outer constructor to create object in the begining.
# Used when models are trained from previous checkpoint
function GANmetaData(
  generator::Chain, discriminator::Chain,
  genOpt::Flux.Optimise.AbstractOptimiser, discOpt::Flux.Optimise.AbstractOptimiser,
  myTrainConfig::trainConfig, metaDataFilepath::String
) 
  genValidations_, discValidations_, testLosses_, _ = getValuesFromTxt(metaDataFilepath)
  return GANmetaData(
    nnInfo(generator, optimisationInfo(genOpt, Optimisers.setup(genOpt, generator)), MVHistory()),
    nnInfo(discriminator, optimisationInfo(discOpt, Optimisers.setup(discOpt, discriminator)), MVHistory()),
    myTrainConfig,
    Dict(
      :genValHistory => Float64.(genValidations_),
      :discValHistory => Float64.(discValidations_),
      :genTest => [Float64(testLosses_[1])],
      :discTest => [Float64(testLosses_[2])]
    ),
    readDataSplits(metaDataFilepath),
    readPercentage(metaDataFilepath), 5
  )
end

function getGen(metaData::GANmetaData)
  return metaData.genDefinition.neuralNetwork
end
function getDisc(metaData::GANmetaData)
  return metaData.discDefinition.neuralNetwork
end

# disable/reenable training in model
function switchTraining(metaData::GANmetaData, mode::Bool)
  Flux.trainmode!(getGen(metaData), mode); Flux.trainmode!(getGen(metaData), mode)
  return nothing
end

# Struct with FEA parameters
@with_kw mutable struct FEAparameters
  quants::Int = 1 # number of TO problems per section
  V::Array{Real} = [0.4 + rand() * 0.5 for i in 1 : quants] # volume fractions
  problems::Any = Array{Any}(undef, quants) # store FEA problem structs
  meshSize::Tuple{Int, Int} = (140, 50) # Size of rectangular mesh
  section::Int = 1 # Number of dataset HDF5 files with "quants" samples each
  nElements::Int32 = prod(meshSize) # quantity of elements
  nNodes::Int32 = prod(meshSize .+ 1) # quantity of nodes
  # matrix with element IDs in their respective position in the mesh
  elementIDmatrix::Array{Int, 2} = convert.(Int, quad(meshSize..., [i for i in 1:nElements]))
  elementIDarray::Array{Int} = [i for i in 1:nElements] # Vector that lists element IDs
  meshMatrixSize::Tuple{Int, Int} = (51, 141) # Size of rectangular nodal mesh as a matrix
end
if runningInColab == false # if running locally
  FEAparams = FEAparameters()
  initializerProblem!(FEAparams)
else
  const FEAparams = FEAparameters()
end

# Squeeze and excitation block for TopologyGAN U-SE-ResNet generator
struct SEblock chain::Chain end
function (m::SEblock)(input) return m.chain(input) end
Flux.@functor SEblock

# Residual block for TopologyGAN U-SE-ResNet generator
struct SEresNet chain::Chain end
function (m::SEresNet)(input) return m.chain(input) end
Flux.@functor SEresNet

# custom Flux.jl split layer
struct Split{T} paths::T end
Split(paths...) = Split(paths)
Flux.@functor Split
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

struct convNext chain::Chain end
function (m::convNext)(x) return m.chain(x) end
Flux.@functor convNext

stochasticDepth(x::Float32) = Flux.Dropout(x; dims = 4)

struct ChannelLayerNorm{D, T} diag::D; ϵ::T end
Flux.@functor ChannelLayerNorm
function ChannelLayerNorm(sz::Integer, λ = identity; ϵ = 1.0f-6)
  diag = Flux.Scale(1, 1, sz, λ)
  return ChannelLayerNorm(diag, ϵ)
end
(m::ChannelLayerNorm)(x) = m.diag(Flux.normalise(x; dims = ndims(x) - 1, ϵ = m.ϵ))

struct addNoise end
Flux.@functor addNoise
# (m::addNoise)(x::AbstractArray) = 