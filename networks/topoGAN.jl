include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
struct SEblock
  chain::Chain
end
function (m::SEblock)(input)
  return m.chain(input)
end
Flux.@functor SEblock
l = SEblock(Chain(
  GlobalMeanPool(),
  flatten,
  Dense(32 => 32 ÷ 16, relu),
  Dense(32 ÷ 16 => 32, sigmoid),
  a -> reshape(a, (1, 1, :, 1))
))
struct U_SE_ResNet
  chain::Chain
end
function (m::U_SE_ResNet)(input)
  return m.chain(input)
end
Flux.@functor U_SE_ResNet
m = U_SE_ResNet(Chain(
  leakyrelu,
  Conv((5, 5), 512 => 512; pad = SamePad()),
  BatchNorm(512),
  leakyrelu,
  Conv((5, 5), 512 => 512; pad = SamePad()),
  BatchNorm(512),
  SkipConnection(l, *),
))
# instantiate architecture
function U_SE_ResNetGenerator(; U_SE_ResNetCount = 32)
  d1e3 = Chain(
    ntuple(i -> SkipConnection(m, +), U_SE_ResNetCount), # Repeat U_SE_ResNet block U_SE_ResNetCount times
    relu,
    ConvTranspose((5, 5), 512 => 512), ### self.d1
    BatchNorm(512), ### d1
  )
  d2e2 = Chain(
    leakyrelu,
    Conv((5, 5), 256 => 512; stride = 2, pad = SamePad()),
    BatchNorm(512), ### e3
    SkipConnection(d1e3, (mx, x) -> cat(mx, x, dims = 3)), # concat d1 e3. ### d1
    relu,
    ConvTranspose((5, 5), 512 => 256; stride = 2), ### self.d2
    BatchNorm(256), ### d2
  )
  d3e1 = Chain(
    leakyrelu,
    Conv((5, 5), 128 => 256; stride = 2, pad = SamePad()),
    BatchNorm(256), ### e2
    SkipConnection(d2e2, (mx, x) -> cat(mx, x, dims = 3)), # concat d2 e2, ### d2
    relu,
    ConvTranspose((5, 5), 256 => 128; stride = 2), ### self.d3
    BatchNorm(128), ### d3
  )
  return Chain(
    Conv((5, 5), 3 => 128; stride = 2, pad = SamePad()), ### e1
    SkipConnection(d3e1, (mx, x) -> cat(mx, x, dims = 3)), # concat d3 e1, ### d3
    relu,
    ConvTranspose((5, 5), 256 => 1; stride = 2), ### self.d4
    sigmoid
  )
end

generator = U_SE_ResNetGenerator() # instantiate architecture
# predict topology from input (vf) and condition (physical fields)
# (input data concatenates physical fields and vf matrices)
topology = generator([data sample])