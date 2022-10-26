include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")

function U_SE_ResNetGenerator()
  Chain(
    Conv((5, 5), 3 => 128, leakyrelu; stride = 2, pad = SamePad()),
    Conv((5, 5), 128 => 256; stride = 2, pad = SamePad()), # NHWC [1, d_h = 2, d_w = 2, 1]
    BatchNorm(256),
    leakyrelu,
    Conv((5, 5), 256 => 512; stride = 2, pad = SamePad()),
    BatchNorm(512),
    # U_SE_ResNet
    # U_SE_ResNet
    # U_SE_ResNet
    # ...
  )
end
function U_SE_ResNet()
  Chain(
    leakyrelu,
    Conv((5, 5), 512 => 512; pad = SamePad()),
    BatchNorm,
    leakyrelu,
    Conv((5, 5), 512 => 512; pad = SamePad()),
    BatchNorm,
    # seblock
  )
end