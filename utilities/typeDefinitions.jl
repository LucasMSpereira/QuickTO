# Custom type definitions

abstract type trainConfig end

# Configurations for training of neural networks for fixed number of epochs
mutable struct epochTrainConfig <: trainConfig
  epochs::Int64 # Total number of training epochs
  validFreq::Int64 # Evaluation frequency in epochs
  schedule::Int64 # Learning rate adjustment interval in epochs. 0 for no scheduling
  decay::Float64 # If scheduling, periodically multiply learning rate by this value
  evaluations::Array{Any} # History of evaluation losses
end
epochTrainConfig(
  epochs, validFreq; schedule = 0, decay = 0.0, evaluations = []
) = epochTrainConfig(epochs, validFreq, schedule, decay, evaluations)

# Configurations for early-stop training of neural networks
mutable struct earlyStopTrainConfig <: trainConfig
  validFreq::Int64 # Evaluation frequency in epochs
  decay::Float64 # If scheduling, periodically multiply learning rate by this value
  schedule::Int32 # Learning rate adjustment interval in epochs. 0 for no scheduling
  evaluations::Array{Any} # History of evaluation losses
  # Interval of most recent validations used for early stop criterion
  earlyStopQuant::Int32
  # Minimal percentage drop in loss in the last "earlyStopQuant" validations
  earlyStopPercent::Float32
  checkPointFreq::Int32 # epoch interval between intermediate saves
end

earlyStopTrainConfig(
  validFreq; decay = 0.0, schedule = 0, evaluations = [],
  earlyStopQuant = 3, earlyStopPercent = 1, checkPointFreq = 10
) = earlyStopTrainConfig(validFreq, decay, schedule, evaluations,
  earlyStopQuant, earlyStopPercent, checkPointFreq
)

# Meta-data for GAN pipeline
mutable struct GANmetaData
  generator::Chain # generator network
  discriminator::Chain # discriminator network
  opt::Flux.Optimise.AbstractOptimiser # optimizer used in training
  loaders::Dict{Symbol, <:MLUtils.DataLoader} # loaders for each dataset split
  const trainConfig::trainConfig # parameters for training
  lossesVals::Dict{Symbol, Vector{Float64}} # loss histories
  files::Dict{Symbol, Vector{String}}
end

## GANmetaData APIs
# save histories of losses
function (ganMD::GANmetaData)(valLossHist::NTuple{2, Float64}; context = :validation)
  if context == :validation
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

# Outer constructor to create object in the begining
GANmetaData(
  generator::Chain, discriminator::Chain, opt::Flux.Optimise.AbstractOptimiser,
  loaders::Dict{Symbol, <:MLUtils.DataLoader}, myTrainConfig::trainConfig
) = GANmetaData(
  generator, discriminator, opt, loaders, myTrainConfig,
  Dict(
    :genValHistory => Float64[],
    :discValHistory => Float64[],
    :genTest => Float64[],
    :discTest => Float64[]
  ),
  Dict(
    :train => String[],
    :validate => String[]
  )
)

getDataset(metaData::GANmetaData, goal::Symbol) = get(metaData.loaders, goal, "")

function switchTraining(metaData::GANmetaData, mode::Bool)
  Flux.trainmode!(metaData.generator, mode); Flux.trainmode!(metaData.discriminator, mode)
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
end
FEAparams = FEAparameters()
problem!(FEAparams)

# Squeeze and excitation block for TopologyGAN U-SE-ResNet generator
struct SEblock
  chain::Chain
end
function (m::SEblock)(input)
  return m.chain(input)
end
Flux.@functor SEblock

# Residual block for TopologyGAN U-SE-ResNet generator
struct SEresNet
  chain::Chain
end
function (m::SEresNet)(input)
  return m.chain(input)
end
Flux.@functor SEresNet

# custom Flux.jl split layer
struct Split{T}
  paths::T
end
Split(paths...) = Split(paths)
Flux.@functor Split
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)