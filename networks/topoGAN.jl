include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")

g = U_SE_ResNetGenerator(); @time rand(Float32, (51, 141, 3, 1)) |> g
function topoGANloss(predTopo, targetTopo)
  # predTopo: batch of outputs from model
  # targetTopo: batch of respective target topologies
  
end