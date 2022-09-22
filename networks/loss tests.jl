include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
### get data
forceData, forceMat, disp, forceMulti = loadDispData("C:/Users/LucasKaoid/Desktop/datasets/data/stressCNNdata/dispData"; multiOut = true)
### separate data for training, validation and test
dispTrainLoader, dispValidateLoader, dispTestLoader = getDisploaders(disp, forceMulti, 0, (0.9, 0.1), 1719; multiOutputs = true)
### Custom loss
rmseLoss(x, y) = sqrt(mean((x .- y) .^ 2)); myMultiLoss(x, y) = multiLoss(x, y; lossFun = rmseLoss)
@load "./networks/models/3-celu-5-2.5854946E1/3-celu-5-2.5854946E1.bson" cpu_model
gpu_model = gpu(cpu_model)
newModel = multiOutputs((3, 3), celu, 5)
trainmode!(gpu_model, false)
trainmode!(newModel, false)
# randSample = rand(1:size(disp, 4))
randSample = 10; @show randSample
input = disp[:, :, :, randSample]; target = [forceMulti[i][:, randSample] for i in axes(forceMulti)[1]]
values = CUDA.zeros(Float32, (11_000, 4)); count = 0
### batchTrain!()
for (x, y) in dispTrainLoader # each batch
  # @show typeof(x); @show size(x); @show typeof(y); @show size(y[1])
  loss = convert.(Float32, [0.0]) |> gpu
  ### myMultiLoss -> multiLoss -> rmseLoss = sqrt(mean((x .- y) .^ 2))
  for sampleInBatch in axes(x)[end] # iterate inside batch
    modelOut = reshape(x[:, :, :, sampleInBatch], (size(x[:, :, :, sampleInBatch])..., 1)) |> gpu |> gpu_model
    truth = Tuple([y[i][:, sampleInBatch] for i in axes(y)[1]]) |> gpu
    count += 1
    for i in axes(modelOut)[1]
      values[count, i] = (modelOut[i] .- truth[i]) .^ 2 |> mean
    end
    # loss = vcat(loss, [(modelOut[i] .- truth[i]) .^ 2 for i in axes(modelOut)[1]] |> mean |> mean |> sqrt)
  end
  # @show loss[2:end] |> mean # -> gradient -> update params
end
values
vals = cpu(values)[1:10313, :]
statsum(vals[:, 1]) # line (175 +- 137)
statsum(vals[:, 2]) # col (620 +- 652)
statsum(vals[:, 3]) # xcomp (913 +- 1295)
statsum(vals[:, 4]) # ycomp (1120 +- 1507)
std(vals[:, 4])
# out = reshape(input, (size(input)..., 1)) |> gpu |> gpu_model |> cpu
# @show out
# printLoss(rmseLoss)
# displayDispPlotTest(FEAparams, input, target, gpu_model, myMultiLoss)