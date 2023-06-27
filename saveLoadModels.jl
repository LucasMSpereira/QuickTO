# script to test saving and loading models in different formats

# since julia rarely has compatibility issues, this project was
# developed accross a handfull of versions of the language (up to
# 1.8.5) without problems. However, the BSON.jl package (used to save model checkpoints)
# is underdeveloped. When switching to Julia 1.9.0, that led to models
# saved using BSON.jl in 1.8.5 not being loaded correctly in Julia 1.9.0.
# To circumvent this, the present script was written in Julia 1.8.5 and uses
# the .bson trained model checkpoints to create .jld2 saves using JLD2.jl,
# a format compatible with Julia 1.9.0.

include("QTOutils.jl")

# use BSON save files to generate jld2 files
begin
  topoGANgen, topoGANdisc = loadTrainedGANs(:topologyGAN, "bson") .|> cpu
  convNextGen, cnCritic = loadTrainedGANs(:convnext, "bson") .|> cpu
  topoGANgenState = Flux.state(topoGANgen)
  topoGANdiscState = Flux.state(topoGANdisc)
  convNextGenState = Flux.state(convNextGen)
  convNextCriticState = Flux.state(cnCritic)
  jldsave(datasetPath * "trainedNetworks/topoGANgen.jld2"; topoGANgenState)
  jldsave(datasetPath * "trainedNetworks/topoGANdisc.jld2"; topoGANdiscState)
  jldsave(datasetPath * "trainedNetworks/convNextGen.jld2"; convNextGenState)
  jldsave(datasetPath * "trainedNetworks/convNextCritic.jld2"; convNextCriticState)
end

# model names and number of input channels
modelChannelTuple = [("topoGANgen", 3), ("topoGANdisc", 7), ("convNextGen", 3), ("convNextCritic", 7)]
# loop in combinations of file formats and models
# to test model loading and usage
# for format in ["jld2" "bson"], (m, inChannel) in modelChannelTuple
for format in ["jld2"], (m, inChannel) in modelChannelTuple
  # instantiate architecture according to case
  m == "topoGANgen" && (model = cpu(U_SE_ResNetGenerator()))
  m == "topoGANdisc" && (model = cpu(topologyGANdisc()))
  m == "convNextGen" && (model = cpu(convNextModel(192, [3, 3, 27, 3], 0.5)))
  m == "convNextCritic" && (model = cpu(convNextCritic(; drop = 0.3)))
  Flux.loadmodel!(model, # load trained model
    JLD2.load(datasetPath * "trainedNetworks/$m.jld2", m * "State")
  )
  model = gpu(model) # send model to gpu
  # test shape of model output
  if m in ["topoGANgen", "convNextGen"]
    @assert model(rand(Float32, (51, 141, inChannel, 256)) |> gpu) |> size == (50, 140, 1, 256)
  elseif m in ["topoGANdisc", "convNextCritic"]
    @assert model(rand(Float32, (51, 141, inChannel, 256)) |> gpu) |> size == (1, 256)
  end
  println("$format $m passed")
end
# test loadTrainedGANs() function
for m in [:topologyGAN, :convnext], format in ["jld2"]
# for m in [:topologyGAN, :convnext], format in ["jld2" "bson"]
  g, d = loadTrainedGANs(m, format)
  @assert g(rand(Float32, (51, 141, 3, 256)) |> gpu) |> size == (50,140, 1, 256)
  @assert d(rand(Float32, (51, 141, 7, 256)) |> gpu) |> size == (1, 256)
  println("$m $format OK")
end