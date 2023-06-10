# script to test saving and loading models in different formats

include("QTOutils.jl")

# model names and number of input channels
modelChannelTuple = [("topoGANgen", 3), ("topoGANdisc", 7), ("convNextGen", 3), ("convNextCritic", 7)]
# loop in combinations of file format and model
for format in ["jld2" "bson"], (m, inChannel) in modelChannelTuple
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

for m in [:topologyGAN, :convnext], format in ["bson", "jld2"]
  g, d = loadTrainedGANs(m, format)
  @assert g(rand(Float32, (51, 141, 3, 256)) |> gpu) |> size == (50,140, 1, 256)
  @assert d(rand(Float32, (51, 141, 7, 256)) |> gpu) |> size == (1, 256)
  println("$m $format OK")
end