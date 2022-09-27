include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
### get data
forceData, forceMat, disp, forceMulti = loadDispData("C:/Users/LucasKaoid/Desktop/datasets/data/stressCNNdata/dispData"; multiOut = true)
### separate data for training, validation and test
dispTrainLoader, dispValidateLoader, dispTestLoader = getDisploaders(disp, forceMulti, 0, (0.9, 0.1), 1719; multiOutputs = true)
### Custom loss
rmseLoss(x, y) = sqrt(mean((x .- y) .^ 2)); myMultiLoss(x, y) = multiLoss(x, y; lossFun = rmseLoss)
@load "./networks/models/3-celu-5-4.1173697E2/3-celu-5-4.1173697E2.bson" cpu_model
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

@load "./networks/models/3-celu-5-4.1173697E2/3-celu-5-4.1173697E2.bson" cpu_model
forceMulti[3] .+= 90
forceMulti[4] .+= 90
statsum(forceMulti[3])
for (disp, force) in dispTrainLoader # each batch
  pred = (rand(1:50, (2, size(force[1], 2))), rand(1:140, (2, size(force[1], 2))),
    rand(0:180, (2, size(force[1], 2))), rand(0:180, (2, size(force[1], 2))))
  batchLoss = []
  for sampleInBatch in 1:size(force[1], 2) # iterate inside batch
    @show sampleInBatch
    for i in axes(pred)[1] # each prediction: i, j, x, y
      @show i
      squaredError = (pred[i][:, sampleInBatch] .- force[i][:, sampleInBatch]) .^ 2
      println(pred[i][:, sampleInBatch], " - ", force[i][:, sampleInBatch], " .^ 2 = ",  squaredError)
      println(".|> mean (MSE): ", mean(squaredError))
    end
    println("|> mean (average MSE in sample): ", [(pred[i][:, sampleInBatch] .- force[i][:, sampleInBatch]) .^ 2 for i in axes(pred)[1]] .|> mean |> mean)
    batchLoss = vcat(batchLoss, [(pred[i][:, sampleInBatch] .- force[i][:, sampleInBatch]) .^ 2 for i in axes(pred)[1]] .|> mean |> mean)
  end
  @show mean(batchLoss)
  println()
  println()
end