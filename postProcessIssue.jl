include("./QTOutils.jl")
dGAN = readTopologyGANdataset(datasetPath * "data/trainValidate2/114 1 766"; print = true);
force, supp, vf, disp, top = getDataFSVDT(datasetPath * "data/trainValidate/114 1 766")
negativeVMsamples = findall(<=(0.0),
  reshape( mapslices( minimum, dGAN[:vm]; dims = [1, 2] ), (:) )
)
length(negativeVMsamples)/size(dGAN[:vm], 4)*100
begin # standard sample plot
  sample = rand(negativeVMsamples); @show sample
  plotSample(FEAparams, supp[:, :, sample], force[:, :, sample], vf[sample],
    top[:, :, sample], disp[:, :, 2 * sample - 1 : 2 * sample], 1, 1, 1
  )
end
begin # heatmap of new VM values
  sample = rand(negativeVMsamples); @show sample
  figg = Figure(resolution = (1400, 700))
  _, hm = heatmap(figg[1, 1],
    1:FEAparams.meshMatrixSize[1], FEAparams.meshMatrixSize[2]:-1:1, dGAN[:vm][:, :, 1, sample]';
    colorrange = (minimum(dGAN[:vm][:, :, 1, sample]), maximum(dGAN[:vm][:, :, 1, sample]))
  )
  Colorbar(figg[1, 2], hm)
  display(figg)
end
dGAN[:vm][:, :, 1, negativeVMsamples] |> statsum
reshape( mapslices( minimum, dGAN[:vm]; dims = [1, 2] ), (:) ) |> statsum
reshape( mapslices( minimum, dGAN[:vm]; dims = [1, 2] ), (:) ) |> scatter
# @time oldVM = [calcVM(FEAparams.nElements, FEAparams, disp[:, :, 2 * s - 1 : 2 * s], 210e3 * vf[s], 0.3) for s in axes(vf, 1)]
# mins = minimum.(oldVM) |> minimum
a = [findall(<(0.0), dGAN[:vm][:, :, 1, s]) for s in negativeVMsamples]
al = length.(a)
f = Figure(resolution = (1400, 700));
ax = Axis(f[1, 1], ylabel = "Amount of negative nodal values", xlabel = "Sample ID")
scatter!(ax, al)
display(f)

force, supp, vf, disp, top = getDataFSVDT(datasetPath * "data/trainValidate/114 1 766")
nodeVM, nodeEnergy, vmEle, energyEle = calcCondsGAN(disp[:, :, 1:2], 210e3 * vf[1], 0.3)
statsum.((nodeVM, nodeEnergy, vmEle, energyEle))
GLMakie.activate!()
begin
  f = Figure(resolution = (1700, 900));
  _, hmNodeVM = heatmap(f[1, 1], 1:FEAparams.meshMatrixSize[1], FEAparams.meshMatrixSize[2]:-1:1, nodeVM';
    colorrange = (floor(Int, nodeVM |> minimum), ceil(Int, nodeVM |> maximum))
  )
  Colorbar(f[1, 2], hmNodeVM)

  _, hmNodeEnergy = heatmap(f[1, 3], 1:FEAparams.meshMatrixSize[1], FEAparams.meshMatrixSize[2]:-1:1, nodeEnergy';
    colorrange = (nodeEnergy |> minimum, nodeEnergy |> maximum)
  )
  Colorbar(f[1, 4], hmNodeEnergy)

  _, hmEleVM = heatmap(f[2, 1], 1:FEAparams.meshSize[2], FEAparams.meshSize[1]:-1:1, vmEle';
    colorrange = (floor(Int, vmEle |> minimum), ceil(Int, vmEle |> maximum))
  )
  Colorbar(f[2, 2], hmEleVM)

  _, hmEleEnergy = heatmap(f[2, 3], 1:FEAparams.meshSize[2], FEAparams.meshSize[1]:-1:1, energyEle';
    colorrange = (energyEle |> minimum, energyEle |> maximum)
  )
  Colorbar(f[2, 4], hmEleEnergy)
  display(f)
end
z = zeros(6, 7)
z[2, 2 : end - 1] .= 2.0
z[end - 1, 2 : end - 1] .= 2.0
z[2 : end - 1, 2] .= 1.0
z[2 : end - 1, end - 1] .= 1.0
interpolation = linear_interpolation((0.5:5.5, 0.5:6.5),
  z, extrapolation_bc = Interpolations.Line()
) # create interpolation object
# (inter/extra)polate centroid values to mesh nodes
interp = interpolation(0:6, 0:7)
_, hm1 = heatmap(0:5, 6:-1:0, z')
_, hm2 = heatmap(0:6, 7:-1:0, interp')