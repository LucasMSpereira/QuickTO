# import Pkg
# Pkg.activate(".")
# Pkg. instantiate()
begin
include("QTOutils.jl")
convNextGen, convNextDisc = loadTrainedGANs(:convnext)
convNextPerf = load("./networks/convNextGen validate.jld2")["single_stored_object"]
plotOutliers(convNextGen, :validation, convNextPerf, "ConvNeXt", :save, 0.995)
end

# Script used to quantify performance of trained models

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
## interpret topologyGAN (explainable AI)
# Generator - average pointwise correlations
corrs = generatorCorrelation(topoGANgen, :validation; additionalFiles = 6)
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
## interpret ConvNeXt (explainable AI)
# Generator - average pointwise correlations
corrs = generatorCorrelation(convNextGen, :validation; additionalFiles = 6)
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