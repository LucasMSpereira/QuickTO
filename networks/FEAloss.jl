#= New training pipeline involves using the forces predicted in FEA.
The loss then compares the displacements from the dataset sample and
the ones resulting from this new analysis. Using actual TopOpt.jl API for
FEA for now, but intend to compare it against PINNs =#

include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
disp, sup, vf = loadFEAlossData() # load data
disp ./= maximum(abs, disp) # normalize in [-1.0; 1.0]
### separate data for training, validation and test
FEAlossTrainLoader, FEAlossValidateLoader, FEAlossTestLoader = getFEAlossLoaders(disp, sup, vf, 100, (0.7, 0.15), 32)
### Loss including FEA
function FEAloss(predForce, trueDisp, sup, vf)
  #= predForce: batch of forces predicted by model
  for samples in current batch:
    trueDisp: true displacements;  sup: mechanical supports;  vf: VF values =#
  batchLoss = []
  for sampleInBatch in 1:size(predForce[1], 2) # iterate inside batch
    # Use predicted forces in FEA to obtain new displacement field
    predDisp = predFEA(Tuple([predForce[i][:, sampleInBatch] for i in 1:4]), vf[sampleInBatch], sup[:, :, sampleInBatch])
    # shift both to [0; 2] range
    predDisp = predDisp ./ maximum(abs, predDisp) .+ 1
    sampleTrueDisp = trueDisp[:, :, :, sampleInBatch] ./ maximum(abs, trueDisp[:, :, :, sampleInBatch]) .+ 1
    batchLoss = vcat(batchLoss, (predDisp - sampleTrueDisp) .^ 2 |> mean) # MSE
  end
  return convert(Float32, mean(batchLoss)) |> gpu
end

# load model. It will be further trained with new FEA loss
@load "./networks/models/5-celu-5-1.0668805E3/5-celu-5-1.0668805E3.bson" cpu_model
@time FEAlossPipeline(gpu(cpu_model), (FEAlossTrainLoader, FEAlossValidateLoader, FEAlossTestLoader),
  FEAparams, FEAloss, Flux.Optimise.NAdam(5e-5), "5-celu-5-1.0668805E3")