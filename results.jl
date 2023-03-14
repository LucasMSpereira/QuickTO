# import Pkg
# Pkg.activate(".")
# Pkg. instantiate()
include("QTOutils.jl")

### TopologyGAN
# load trained topologyGAN models
topoGANgen, topoGANdisc = loadTrainedGANs(:topologyGAN)
# training and validation data splits used for ConvNeXt generator
# fileSplit = readDataSplits("./networks/GANplots/reuniao7-10.0%-2-12-12T19-00-34/12-12T00-00-13metaData.txt")
# # topology, VF and compliance errors in data splits
# splitGoal = :test
# # resultDict = genPerformance(topoGANgen, fileSplit[splitGoal])
# resultDict = genPerformance(topoGANgen, [datasetPath * "data/test"])
# save_object("./networks/topoGAN $(String(splitGoal)).jld2", resultDict)
# topoGANperf = load("./networks/topoGAN test.jld2")["single_stored_object"]
# statsum(topoGANperf[:topoSE])
# statsum(topoGANperf[:VFerror])
# statsum(topoGANperf[:compError])
# topoGANCompError = filter(<(quantile(topoGANperf[:compError], 0.992)), topoGANperf[:compError])
# statsum(topoGANCompError)
# interpret topologyGAN (explainable AI)
genShapleyValPlots(:training, topoGANgen, "asd")
# plot results from generator
trainedSamples(10, 5, topoGANgen, "topoGAN"; split = :training)
trainedSamples(10, 5, topoGANgen, "topoGAN"; split = :validation)
trainedSamples(10, 5, topoGANgen, "topoGAN"; split = :test)

### ConvNeXt
# load trained ConvNeXt models
convNextGen, convNextDisc = loadTrainedGANs(:convnext)
# training and validation data splits used for ConvNeXt generator
fileSplit = readDataSplits("./networks/GANplots/01-29T09-45-03-Bvp4/01-29T20-07-39metaData.txt")
splitGoal = :test
# resultDict = genPerformance(convNextGen, fileSplit[splitGoal])
resultDict = genPerformance(convNextGen, [datasetPath * "data/test"])
save_object("./networks/convNextGen $(String(splitGoal)).jld2", resultDict)
convNextPerf = load("./networks/convNextGen test.jld2")["single_stored_object"]
statsum(convNextPerf[:topoSE])
statsum(convNextPerf[:VFerror])
statsum(convNextPerf[:compError])
quantile(convNextPerf[:compError], 0.916)
convNextCompError = filter(<(quantile(convNextPerf[:compError], 0.916)), convNextPerf[:compError])
statsum(convNextCompError)
# plot results from generator
trainedSamples(10, 5, convNextGen, "convnext"; split = :training)
trainedSamples(10, 5, convNextGen, "convnext"; split = :validation)
trainedSamples(10, 5, convNextGen, "convnext"; split = :test)