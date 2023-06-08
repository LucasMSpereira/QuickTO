# Script used to quantify performance of trained models
begin
  include("QTOutils.jl")
  topoGANgen, topoGANdisc = loadTrainedGANs(:topologyGAN, "bson") .|> cpu
  convNextGen, convNextDisc = loadTrainedGANs(:convnext, "bson") .|> cpu
  topoGANgenState = Flux.state(topoGANgen)
  topoGANdiscState = Flux.state(topoGANdisc)
  convNextGenState = Flux.state(convNextGen)
  convNextDiscState = Flux.state(convNextDisc)
  jldsave(datasetPath * "trainedNetworks/topoGANgenState.jld2"; topoGANgenState)
  jldsave(datasetPath * "trainedNetworks/topoGANdiscState.jld2"; topoGANdiscState)
  jldsave(datasetPath * "trainedNetworks/convNextGenState.jld2"; convNextGenState)
  jldsave(datasetPath * "trainedNetworks/convNextDiscState.jld2"; convNextDiscState)
end
begin
  # model = cpu(convNextModel(192, [3, 3, 27, 3], 0.5))
  model = cpu(topologyGANdisc(; drop = 0.3))
  Flux.loadmodel!(
    model,
    JLD2.load(datasetPath * "trainedNetworks/convNextDiscState.jld2", "convNextDiscState")
  )
  model = gpu(model)
  model(rand(Float32, (51, 141, 7, 256)) |> gpu)
end

### TopologyGAN
# load trained topologyGAN models
topoGANgen, topoGANdisc = loadTrainedGANs(:topologyGAN)
# training and validation data splits used for U-SE-ResNet generator
fileSplit = readDataSplits("./networks/GANplots/reuniao7-10.0%-2-12-12T19-00-34/12-12T00-00-13metaData.txt")
## topology, VF and compliance errors in data splits
splitGoal = :validation
# if training or validation split
# resultDict = genPerformance(topoGANgen, fileSplit[splitGoal])
# if test split
resultDict = genPerformance(topoGANgen, [datasetPath * "data/test"])
# save/load results
save_object("./networks/topoGAN $(String(splitGoal)).jld2", resultDict)
topoGANperf = load("./networks/topoGAN validate.jld2")["single_stored_object"]
# statistical summary of results
statsum(topoGANperf[:topoSE])
statsum(topoGANperf[:VFerror])
statsum(topoGANperf[:compError])
# compliance error quantile (looking for outliers)
topoGANCompError = filter(<(quantile(topoGANperf[:compError], 0.992)), topoGANperf[:compError])
# statistical summary of compliance error without outliers
statsum(topoGANCompError)
# plot including examples of outliers in each error metric
plotOutliers(topoGANgen, :validation, topoGANperf, "U-SE-ResNet", :save, 0.995)
## interpret topologyGAN (explainable AI)
# Generator - average pointwise correlations
corrs = generatorInterpretation(topoGANgen, :validation, :pixelwise)
[@show mean(corr) for corr in corrs]
# Discriminator - (input * gradient) for each channel
# input data
data = dataBatch(:test, 1)[2:4]
# reshape input
input = eachslice(cat(data...; dims = 3); dims = 4) |> collect .|> Array .|> dim4
# create analyzer and plot interpretations of input channels
analyzer = ExplainableAI.InputTimesGradient(cpu(topoGANdisc))
explainDiscCritic(rand(input), analyzer, "topologyGAN"; goal = :save)
## plot examples (input, von Mises, suggested and target topologies)
trainedSamples(10, 5, topoGANgen, "topoGAN"; split = :training)
trainedSamples(10, 5, topoGANgen, "topoGAN"; split = :validation)
trainedSamples(10, 5, topoGANgen, "topoGAN"; split = :test)

### ConvNeXt
# load trained ConvNeXt models (QuickTO)
convNextGen, convNextDisc = loadTrainedGANs(:convnext)
# training and validation data splits used for ConvNeXt generator
fileSplit = readDataSplits("./networks/GANplots/01-29T09-45-03-Bvp4/01-29T20-07-39metaData.txt")
## topology, VF and compliance errors in data splits
splitGoal = :test
# if training or validation split
# resultDict = genPerformance(convNextGen, fileSplit[splitGoal])
# if test split
resultDict = genPerformance(convNextGen, [datasetPath * "data/test"])
# save/load results
save_object("./networks/convNextGen $(String(splitGoal)).jld2", resultDict)
convNextPerf = load("./networks/convNextGen test.jld2")["single_stored_object"]
# statistical summary of results
statsum(convNextPerf[:topoSE])
statsum(convNextPerf[:VFerror])
statsum(convNextPerf[:compError])
# compliance error quantile (looking for outliers)
convNextCompError = filter(<(quantile(convNextPerf[:compError], 0.916)), convNextPerf[:compError])
# statistical summary of compliance error without outliers
statsum(convNextCompError)
# plot including examples of outliers in each error metric
plotOutliers(convNextGen, :validation, convNextPerf, "ConvNeXt", :save, 0.995)
## interpret ConvNeXt (explainable AI)
# Generator - average pointwise correlations
corrs = generatorInterpretation(convNextGen, :validation, :pixelwise)
[@show mean(corr) for corr in corrs]
# Discriminator - (input * gradient) for each channel
# input data
data = dataBatch(:test, 1)[2:4]
# reshape input
input = eachslice(cat(data...; dims = 3); dims = 4) |> collect .|> Array .|> dim4
# create analyzer and plot interpretations of input channels
analyzer = ExplainableAI.InputTimesGradient(cpu(convNextDisc))
explainDiscCritic(rand(input), analyzer, "QuickTO"; goal = :save)
## plot examples (input, von Mises, suggested and target topologies)
trainedSamples(10, 5, convNextGen, "convnext"; split = :training)
trainedSamples(10, 5, convNextGen, "convnext"; split = :validation)
trainedSamples(10, 5, convNextGen, "convnext"; split = :test)

## Compare computational performance
import Nonconvex
Nonconvex.@load NLopt
const to = TimerOutput()
reset_timer!(to)
sampleAmount = 150
methodThroughput(sampleAmount, loadTrainedGANs(:topologyGAN)[1], loadTrainedGANs(:convnext)[1])
show(to)
standardTOseconds = TimerOutputs.time(to["standard"])/sampleAmount/1e9
println()
println(round(standardTOseconds; digits = 1), " s")
uSEresNetSeconds = TimerOutputs.time(to["U-SE-ResNet"])/sampleAmount/1e9
println("$(round(uSEresNetSeconds; digits = 1)) s ($(round(uSEresNetSeconds/standardTOseconds * 100; digits = 1))%)")
quickTOseconds = TimerOutputs.time(to["QuickTO"])/sampleAmount/1e9
println("$(round(quickTOseconds; digits = 1)) s ($(round(quickTOseconds/standardTOseconds * 100; digits = 1))%)")

# generator interpretation
for net in [:topologyGAN, :convnext], p in [2, 3]
  corrs = generatorInterpretation(
    loadTrainedGANs(net, "bson")[1], :validation, :binaryPerturb;
    additionalFiles = 8, perturbation = 0.5, perturbedChannel = p
  )
end