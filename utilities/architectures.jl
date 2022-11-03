# Builders for different NN architectures

# Discriminator for TopologyGAN
# https://arxiv.org/abs/2003.04685
function discriminator()
  m1 = Chain(
    Conv((5, 5), 6 => df_dim; stride = 2, pad = SamePad()),
    leakyrelu, # h0
    Conv((5, 5), df_dim => df_dim * 2; stride = 2, pad = SamePad()),
    BatchNorm(df_dim * 2),
    leakyrelu, # h1
    Conv((5, 5), df_dim * 2 => df_dim * 4; stride = 2, pad = SamePad()),
    BatchNorm(df_dim * 4),
    leakyrelu, # h2
    Conv((5, 5), df_dim * 4 => df_dim * 8; stride = 2, pad = SamePad()),
    BatchNorm(df_dim * 8),
    leakyrelu, # h3
    flatten,
  )
  m1size = prod(Flux.outputsize(m1, (51, 141, 6, 1)))
  m2 = Split(
    Chain(Dense(m1size => 1, sigmoid), x -> reduce(*, x)),
    Chain(Dense(m1size => 1), x -> reduce(*, x)),
  )
  return Chain(m1, m2)
end

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
    aa -> reshape(aa, (1, 1, :, 1))
  )
  se = SEblock(Chain(se1, se2)) # squeeze and excitation (SE) block
  seRes = SEresNet(Chain(
    leakyrelu,
    Conv((5, 5), gf_dim * 4 => gf_dim * 4; pad = SamePad()),
    BatchNorm(gf_dim * 4),
    leakyrelu,
    Conv((5, 5), gf_dim * 4 => gf_dim * 4; pad = SamePad()),
    BatchNorm(gf_dim * 4),
    SkipConnection(
      se,
      # SE uses global avg pooling to define weights for tensor channels
      (channelWeights, inputTensor) -> inputTensor .* reshape(channelWeights, (1, 1, :))
    ),
  ))
  return Chain(ntuple(i -> SkipConnection(seRes, +), sizeChain))
end

# U-SE-ResNet generator from original article
# https://arxiv.org/abs/2003.04685
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
    Conv((5, 5), 3 => gf_dim; stride = 2, pad = (8, 7)), ### e1
    SkipConnection(d3e1, (mx, x) -> cat(mx, x, dims = 3)), # concat d3 e1, ### d3
    relu,
    ConvTranspose((5, 5), gf_dim * 2 => 1; stride = 2, pad = SamePad()), ### self.d4
    MeanPool((15, 13); stride = 1),
    sigmoid,
  )
end