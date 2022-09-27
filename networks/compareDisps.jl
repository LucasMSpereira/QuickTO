# Compare displacements from dataset against those resulting from the forces predicted by loadCNN:
# dataDisplacement -> loadCNN -> predictedForces (+ BCs + material) -> FEA -> predDisplacements -> vs. dataDisplacement
include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
using GLMakie
GLMakie.activate!()
# * model may have been trained on data that went through preparation *
@load "./networks/models/3-celu-5-4.1173697E2/3-celu-5-4.1173697E2.bson" cpu_model # load model
begin
  ### choose folder, file and sample, get info and make load prediction ###
  id = string(rand(1:6)) # folder
  fileList = glob("*", "C:/Users/LucasKaoid/Desktop/datasets/data/"*id)
  file = rand(fileList)
  force, supp, vf, disp, _ = getDataFSVDT(file) # read data from current hdf5 file
  while true
    sample = rand(axes(vf)[1]) # iterate in samples from current file
    if supp[1, 3, sample] == 4 # skip iteration if left side not clamped
      dataDisp = unsqueeze(disp[:, :, 2*sample - 1 : 2*sample]; dims = 4) |> x -> convert.(Float32, x) # sample displacements
      predForce = dataDisp |> cpu_model .|> x -> convert.(Float64, x) # forces predicted by load CNN model
      replace!(x -> min(FEAparams.meshSize[1], x), predForce[2])
      replace!(x -> max(1, x), predForce[2])
      replace!(x -> min(FEAparams.meshSize[2], x), predForce[1])
      replace!(x -> max(1, x), predForce[1])
      predForce = reduce(hcat, [predForce[i] for i in axes(predForce)[1]]) # reshape predicted forces
      [println(round.(predForce[l, :]; digits = 2)) for l in axes(predForce)[1]]
      ### load prediction -> problem definition -> FEA -> displacements
        # nodeCoords, cells = mshData(FEAparams.meshSize)
        # nodeSets, dispBC = simplePins!("rand", dispBC, FEAparams)
        # problem = InpStiffness(InpContent(
        #   nodeCoords, "CPS4", cells, nodeSets,
        #   Dict("SolidMaterialSolid" => FEAparams.elementIDarray, # cellsets
        #         "Eall"               => FEAparams.elementIDarray,
        #         "Evolumes"           => FEAparams.elementIDarray),
        #   FEAparams.V[i]*210e3, 0.3, 0.0,
        #   Dict("supps" => [(1, 0.0), (2, 0.0)]),
        #   cLoads,
        #   Dict("uselessFaces" => [(1,1)]), Dict("uselessFaces" => 0.0)))
      solver = FEASolver(Direct, rebuildProblem(vf[sample], supp[:, :, sample], predForce);
        xmin = 1e-6, penalty = TopOpt.PowerPenalty(3.0)) # FEA solver
      solver() # determine displacements
      feaDisp = copy(solver.u) # vector with displacements resulting from predicted forces
      # reshape result from FEA
      predDisp = [quad(FEAparams.meshSize .+ 1..., [feaDisp[i] for i in 1:2:length(feaDisp)]);;;
      quad(FEAparams.meshSize .+ 1..., [feaDisp[i] for i in 2:2:length(feaDisp)])]
      ### Plot both von Mises fields ###
      fig = Figure(resolution = (1400, 700), fontsize = 20)
      Label(fig[1, 1], "Data", tellheight = false); Label(fig[2, 1], "Predicted", tellheight = false) # labels
      plotVM(FEAparams, dataDisp, vf[sample], fig, [1, 2]); plotVM(FEAparams, predDisp, vf[sample], fig, [2, 2])
      display(fig)
      colsize!(fig.layout, 1, Fixed(200))
      ### Compare FEA result against original displacements ###
      # statistical summaries
      println("\ndataDisp"); statsum(dataDisp)
      println("\npredDisp"); statsum(predDisp)
      # nodal displacement norm errors
      println("\nNodal norms")
      dataNorms = mapslices(norm, dataDisp; dims = [3])
      predNorms = mapslices(norm, predDisp; dims = [3])
      RMSEnorms = (predNorms - dataNorms) .^ 2 |> mean |> sqrt
      println("RMSE: ", sciNotation(RMSEnorms, 4))
      MAEnorms = abs.(predNorms - dataNorms) |> mean
      println("MAE: ", sciNotation(MAEnorms, 4))
      # nodal displacement component errors
      println("\nDisplacement components")
      RMSEcomps = [(predDisp[:, :, i] - dataDisp[:, :, i]) .^ 2 |> mean |> sqrt for i in axes(predDisp)[3]]
      println("RMSE X: ", sciNotation(RMSEcomps[1], 4), "    RMSE Y: ", sciNotation(RMSEcomps[2], 4))
      MAEcomps = [(predDisp[:, :, i] - dataDisp[:, :, i]) .|> abs |> mean for i in axes(predDisp)[3]]
      println("MAE X: ", sciNotation(MAEcomps[1], 4), "    MAE Y: ", sciNotation(MAEcomps[2], 4))
      break
    end
  end
end