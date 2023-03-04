include("QTOutils.jl")

# load models from topologyGAN
topoGANgen, topoGANdisc = loadTrainedGANs(:topologyGAN)
# plot results from generator
trainedSamples(10, 5, topoGANgen, "topoGAN"; split = :validation)

# load models from ConvNeXt
convNextGen, convNextDisc = loadTrainedGANs(:convnext)
# plot results from generator
trainedSamples(10, 5, convNextGen, "convnext"; split = :training)