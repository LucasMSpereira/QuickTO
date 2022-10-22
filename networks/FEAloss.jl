#= New training pipeline involves using the forces predicted in FEA.
The loss then compares the displacements from the dataset sample and
the ones resulting from this new analysis. Using actual TopOpt.jl API for
FEA for now, but intend to compare it against PINNs =#

include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
disp, sup, vf, force = loadFEAlossData() # load data
disp ./= maximum(abs, disp) .+ 1 # normalize and shift from [-1; 1] to [0; 2]
force[3] .+= 90; force[4] .+= 90 # shift force components from [-90; 90] to [0; 180]
### separate data for training, validation and test
FEAlossTrainLoader, FEAlossValidateLoader, FEAlossTestLoader = getFEAlossLoaders(disp, sup, vf, force, 1000, (0.7, 0.15), 32)
### Loss including FEA
function FEAloss(predForce, trueDisp, sup, vf, trueForce)
  #= predForce: batch of forces predicted by model
  for samples in current batch:
    trueDisp: true displacements;  sup: mechanical supports;
    vf: VF values;  trueForce: forces from dataset =#
  batchDispError = Float32[]
  for sampleInBatch in 1:size(predForce[1], 2) # iterate inside batch
    # Use predicted forces in FEA to obtain new displacement field
    predDisp = predFEA(Tuple([predForce[i][:, sampleInBatch] for i in 1:4]),
      vf[sampleInBatch], sup[:, :, sampleInBatch])
    predDisp = predDisp ./ maximum(abs, predDisp) .+ 1 # normalize and shift from [-1; 1] to [0; 2]
    dispError = (predDisp - trueDisp[:, :, :, sampleInBatch]) .^ 2 |> mean # displacement error. MSE
    batchDispError = vcat(batchDispError, dispError)
  end
  forceError = predForce .- trueForce .|> x -> x .^ 2 .|> mean |> mean # force error. MSE
  return convert(Float32, mean(batchDispError)*1000 + mean(forceError))
end
# load model. It will be further trained with new FEA loss
@load "./networks/models/5-celu-5-1.0668805E3/5-celu-5-1.0668805E3.bson" cpu_model
gpu_model = gpu(cpu_model)
@time FEAlossPipeline(gpu_model, (FEAlossTrainLoader, FEAlossValidateLoader, FEAlossTestLoader),
  FEAparams, FEAloss, Flux.Optimise.NAdam(5e-5), "5-celu-5-1.0668805E3")
#