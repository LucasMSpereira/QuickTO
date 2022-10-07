include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
### get data
forceData, forceMat, disp, forceMulti = loadDispData(true)
disp ./= maximum(abs, disp) # normalize in [-1.0; 1.0]
forceMulti[3] .+= 90; forceMulti[4] .+= 90 # shift force components from [-90; 90] to [0; 180]
### separate data for training, validation and test
dispTrainLoader, dispValidateLoader, dispTestLoader = getDisploaders(disp, forceMulti, 10, (0.7, 0.15), 7; multiOutputs = true)
# Custom loss function for mulitple outputs in predicting loads from displacements
function customLoss2(x, y)
  # x: model output from one batch, y: batch of labels
  batchLoss = []
  for sampleInBatch in 1:size(y[1], 2) # iterate inside batch
    samplePred = [x[1][:, sampleInBatch];; x[2][:, sampleInBatch];; x[3][:, sampleInBatch] .- 90;; x[4][:, sampleInBatch] .- 90]
    sampleTrue = [y[1][:, sampleInBatch];; y[2][:, sampleInBatch];; y[3][:, sampleInBatch] .- 90;; y[4][:, sampleInBatch] .- 90]
    # position error
    [println(samplePred[f, :]) for f in 1:2]
    xMat = zeros(Float32, (2, 4))
    [xMat[:, i] = samplePred[i] for i in axes(samplePred)[1]]
    xMat =  reduce(hcat, samplePred); [println(xMat[f, :]) for f in 1:2]
    yMat =  reduce(hcat, sampleTrue); [println(yMat[f, :]) for f in 1:2]
    @show [(samplePred[i] - sampleTrue[i]).^2 for i in 1:2]
    posError = [(samplePred[i] - sampleTrue[i]).^2 for i in 1:2] |> sum |> sum
    @show posError
    # alignment
    xyPredFirst = [samplePred[i][1] for i in 3:4]; @show xyPredFirst
    xyPredSecond = [samplePred[i][2] for i in 3:4]; @show xyPredSecond
    xyTrueFirst = [sampleTrue[i][1] for i in 3:4]; @show xyTrueFirst
    xyTrueSecond = [sampleTrue[i][2] for i in 3:4]; @show xyTrueSecond
    alFirst = dot(xyPredFirst/norm(xyPredFirst), xyTrueFirst/norm(xyTrueFirst))
    alSecond = dot(xyPredSecond/norm(xyPredSecond), xyTrueSecond/norm(xyTrueSecond))
    alignmentError = alFirst + alSecond; @show alignmentError
    # norm
    firstNormError = 1 - norm(xyPredFirst)/norm(xyTrueFirst)
    secondNormError = 1 - norm(xyPredSecond)/norm(xyTrueSecond)
    normError = norm([firstNormError secondNormError]); @show normError
    batchLoss = vcat(batchLoss, posError - alignmentError + normError)
  end
  return convert(Float32, mean(batchLoss)) |> gpu
end
m = multiOutputs((5, 5), celu, 5)
batchTrain!(dispTrainLoader, m, Flux.Optimise.NAdam(8e-5), customLoss2)
## Grid search for architecture hyperparameters
# hyperGrid(multiOutputs, [5], [celu], [5], (dispTrainLoader, dispValidateLoader, dispTestLoader),
#   FEAparams, customLoss2, Flux.Optimise.NAdam(8e-5); multiLossArch = true, earlyStop = 1)
#