# Builders for different NN architectures

# loadCNN structure 14. Predict positions and components of loads from displacement field
function multiOutputs(kernel, activ, ch)
  module1 = Chain(
    BatchNorm(2),
    Conv(kernel, 2 => 2, activ),
    BatchNorm(2),
    Conv(kernel, 2 => 2, activ),
    BatchNorm(2),
    Conv(kernel, 2 => 2, activ),
    BatchNorm(2),
    Conv(kernel, 2 => 2, activ),
    BatchNorm(2),
    Conv(kernel, 2 => 2, activ),
    BatchNorm(2),
    Conv(kernel, 2 => 1, activ),
    BatchNorm(1),
    Conv(kernel, 1 => 1, activ),
    BatchNorm(1),
    Conv(kernel, 1 => 1, activ),
    BatchNorm(1),
    Conv(kernel, 1 => 1, activ),
    flatten)
  m1size = prod(Flux.outputsize(module1, (51, 141, 2, 1)))
  module2 = Split(
    Chain(Dense(m1size => m1size ÷ 10), Dense(m1size ÷ 10 => 2, activ)),
    Chain(Dense(m1size => m1size ÷ 10), Dense(m1size ÷ 10 => 2, activ)),
    Chain(Dense(m1size => m1size ÷ 10), Dense(m1size ÷ 10 => 2, activ)),
    Chain(Dense(m1size => m1size ÷ 10), Dense(m1size ÷ 10 => 2, activ)),
  )
  return Chain(module1, module2) |> gpu
end

# custom split layer
struct Split{T}
  paths::T
end
Split(paths...) = Split(paths)
Flux.@functor Split
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

# Squeeze and excitation block for TopologyGAN U-SE-ResNet generator
struct SEblock
  chain::Chain
end
function (m::SEblock)(input)
  return m.chain(input)
end
Flux.@functor SEblock

# Residual block for TopologyGAN U-SE-ResNet generator
struct SEresNet
  chain::Chain
end
function (m::SEresNet)(input)
  return m.chain(input)
end
Flux.@functor SEresNet

# Return Flux Chain of SE-ResNet blocks of desired size.
# Generator from original paper used 32 blocks
function SE_ResNetChain(; sizeChain = 32)
  se1 = Chain(GlobalMeanPool(), flatten)
  se1Size = prod(Flux.outputsize(se1, (1, 1, gf_dim * 4, 1)))
  se2 = Chain(
    Dense(se1Size => se1Size ÷ 16, relu),
    Dense(se1Size ÷ 16 => se1Size, sigmoid),
    a -> reshape(a, (1, 1, :, 1))
  )
  se = SEblock(Chain(se1, se2))
  seRes = SEresNet(Chain(
    leakyrelu,
    Conv((5, 5), gf_dim * 4 => gf_dim * 4; pad = SamePad()),
    BatchNorm(gf_dim * 4),
    leakyrelu,
    Conv((5, 5), gf_dim * 4 => gf_dim * 4; pad = SamePad()),
    BatchNorm(gf_dim * 4),
    SkipConnection(
      se,
      (channelWeights, inputTensor) -> SEmult(channelWeights, inputTensor)
    ),
  ))
  return Chain(ntuple(i -> SkipConnection(seRes, +), sizeChain))
end