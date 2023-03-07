include("QTOutils.jl")

# load trained topologyGAN models
topoGANgen, topoGANdisc = loadTrainedGANs(:topologyGAN)
# training and validation data splits used for ConvNeXt generator
fileSplit = readDataSplits("./networks/GANplots/reuniao7-10.0%-2-12-12T19-00-34/12-12T00-00-13metaData.txt")
resultDict = genPerformance(topoGANgen, fileSplit[:train])
# plot results from generator
trainedSamples(10, 5, topoGANgen, "topoGAN"; split = :training)
trainedSamples(10, 5, topoGANgen, "topoGAN"; split = :validation)
trainedSamples(10, 5, topoGANgen, "topoGAN"; split = :test)

# load trained ConvNeXt models
convNextGen, convNextDisc = loadTrainedGANs(:convnext)
# training and validation data splits used for ConvNeXt generator
fileSplit = readDataSplits("./networks/GANplots/01-29T09-45-03-Bvp4/01-29T20-07-39metaData.txt")
resultDict = genPerformance(convNextGen, fileSplit[:train])
# plot results from generator
trainedSamples(10, 5, convNextGen, "convnext"; split = :training)
trainedSamples(10, 5, convNextGen, "convnext"; split = :validation)
trainedSamples(10, 5, convNextGen, "convnext"; split = :test)