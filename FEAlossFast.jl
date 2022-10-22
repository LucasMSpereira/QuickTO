include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
disp, sup, vf, force = loadFEAlossData(); disp ./= maximum(abs, disp) .+ 1; force[3] .+= 90; force[4] .+= 90
tl, FEAlossValidateLoader, FEAlossTestLoader = getFEAlossLoaders(disp, sup, vf, force, 10, (0.7, 0.15), 5)
# feasolver
function bfast(trainDataLoader, mlModel, opt)
  for (trueDisps, sup, vf, force) in trainDataLoader # each batch
    gTime = 0; grads = 0
    gTime = @elapsed grads = Flux.gradient(() -> FEAlossFast(trueDisps |> gpu |> mlModel |> cpu, trueDisps, sup, vf, force), Flux.params(mlModel))
    println("Gradient(FEAlossFast): ", sciNotation(gTime, 2))
    # Optimization step for current batch
    upTime = @elapsed Flux.Optimise.update!(opt, Flux.params(mlModel), grads)
    println("Gradient update: ", sciNotation(upTime, 2))
    break
  end
end
function FEAlossFast(predForce, trueDisp, sup, vf, trueForce)
  batchDispError = Float32[]
  predFEAtime, dispNormTime, dispErrorTime = 0.0, 0.0, 0.0
  for sampleInBatch in axes(predForce[1])[2] # iterate inside batch
    # Use predicted forces in FEA to obtain new displacement field
    predFEAtime += @elapsed predDisp = predFEA(Tuple([predForce[i][:, sampleInBatch] for i in 1:4]), vf[sampleInBatch], sup[:, :, sampleInBatch])
    dispNormTime += @elapsed predDisp = predDisp ./ maximum(abs, predDisp) .+ 1 # normalize and shift from [-1; 1] to [0; 2]
    dispErrorTime += @elapsed dispError = (predDisp - trueDisp[:, :, :, sampleInBatch]) .^ 2 |> mean # displacement error. MSE
    batchDispError = vcat(batchDispError, dispError)
  end
  forceErrorTime = @elapsed forceError = predForce .- trueForce .|> x -> x .^ 2 .|> mean |> mean # force error. MSE
  @ignore_derivatives println("predFEA: ", sciNotation(predFEAtime, 2))
  @ignore_derivatives println("dispNorm: ", sciNotation(dispNormTime, 2))
  @ignore_derivatives println("dispError: ", sciNotation(dispErrorTime, 2))
  @ignore_derivatives println("forceError: ", sciNotation(forceErrorTime, 2))
  @ignore_derivatives println("FEAlossFast: ", sciNotation(sum([predFEAtime dispNormTime dispErrorTime forceErrorTime]), 2))
  return convert(Float32, mean(batchDispError)*1000 + mean(forceError))
end
@load "./networks/models/5-celu-5-1.0668805E3/5-celu-5-1.0668805E3.bson" cpu_model
model = gpu(cpu_model)
optt = Flux.Optimise.NAdam(5e-5)
for i in 1:2
  Profile.clear()
  @profile bfast(tl, model, optt)
  Profile.print()
end