include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
# instantiate architecture

function U_SE_ResNetGenerator()
  seResNetblocks = SE_ResNetChain()
  d1e3 = Chain(
    seResNetblocks, # Long chain of SE-Res-Net blocks
    relu,
    ConvTranspose((5, 5), gf_dim * 4 => gf_dim * 4, pad = SamePad()), ### self.d1
    BatchNorm(gf_dim * 4), ### d1
  )
  d2e2 = Chain(
    leakyrelu,
    Conv((5, 5), gf_dim * 2 => gf_dim * 4; stride = 2, pad = SamePad()),
    BatchNorm(gf_dim * 4), ### e3
    SkipConnection(d1e3, (mx, x) -> cat(mx, x, dims = 3)), # concat d1 e3. ### d1
    relu,
    ConvTranspose((5, 5), gf_dim * 8 => gf_dim * 2; stride = 2, pad = SamePad()), ### self.d2
    BatchNorm(gf_dim * 2), ### d2
  )
  d3e1 = Chain(
    leakyrelu,
    Conv((5, 5), gf_dim => gf_dim * 2; stride = 2, pad = SamePad()),
    BatchNorm(gf_dim * 2), ### e2
    SkipConnection(d2e2, (mx, x) -> cat(mx, x, dims = 3)), # concat d2 e2, ### d2
    relu,
    ConvTranspose((5, 5), gf_dim * 4 => gf_dim; stride = 2, pad = SamePad()), ### self.d3
    BatchNorm(gf_dim), ### d3
  )
  return Chain(
    Conv((5, 5), 3 => gf_dim; stride = 2, pad = (5, 4)), ### e1
    SkipConnection(d3e1, (mx, x) -> cat(mx, x, dims = 3)), # concat d3 e1, ### d3
    relu,
    # ConvTranspose((5, 5), gf_dim * 2 => 1; stride = 2), ### self.d4
    ConvTranspose((5, 5), gf_dim * 2 => 1; stride = 2, pad = SamePad()), ### self.d4
    Conv((7, 5), 1 => 1; stride = 1),
    sigmoid,
  )
end
generator = U_SE_ResNetGenerator() # instantiate architecture
# predict topology from input (vf) and condition (physical fields)
# (input data concatenates physical fields and vf matrices)
rand(Float32, (50, 140, 3, 1)) |> generator
function plotTopoPred(originalTopo, predTopo; goal = "save")
  fig = Figure(resolution = (1100, 800)) # create makie figure
  # labels for each heatmap
  Label(fig[1, 1], "Original", textsize = 20; tellheight = false); Label(fig[2, 1], "Prediction", textsize = 20; tellheight = false)
  _, hmPred = heatmap( # heatmap of predicted topology
    fig[2, 2], 1:FEAparams.meshSize[2], FEAparams.meshSize[1]:-1:1,
    Array(reshape(predTopo, (FEAparams.meshSize[2], FEAparams.meshSize[1]))')
  )
  colsize!(fig.layout, 2, Fixed(800))
  _, hmOriginal = heatmap( # heatmap of original topology
    fig[1, 2], 1:FEAparams.meshSize[2], FEAparams.meshSize[1]:-1:1,
    Array(originalTopo')
  )
  # colorbars
  Colorbar(fig[1, 3], hmPred); Colorbar(fig[2, 3], hmOriginal)
  if goal == "display" # display image
    GLMakie.activate!()
    display(fig)
  elseif goal == "save" # save image
    Makie.save("./networks/test.pdf", fig)
  else
    error("Invalid kw arg 'goal' in plotTopoPred().")
  end
end
plotTopoPred(rand(Float32, (50, 140)), rand(Float32, (50, 140, 3, 1)) |> generator)