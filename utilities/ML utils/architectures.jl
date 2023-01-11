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

# Discriminator for TopologyGAN
# https://arxiv.org/abs/2003.04685
function topologyGANdisc()
  m1 = Chain(
    Conv((5, 5), 7 => df_dim; stride = 2, pad = SamePad()),
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
  m1size = prod(Flux.outputsize(m1, (51, 141, 7, 1)))
  m2 = Chain(
    Dense(m1size => 1) # h4 (don't need sigmoid)
  )
  return Chain(m1, m2) |> gpu
end

# Return Flux Chain of SE-ResNet blocks of desired size.
# Generator from original paper used 32 blocks
function SE_ResNetChain(sizeChain)
  se1 = Chain(GlobalMeanPool(), flatten)
  se1Size = prod(Flux.outputsize(se1, (1, 1, gf_dim * 4, 1)))
  se2 = Chain(
    Dense(se1Size => se1Size ÷ 16, relu),
    Dense(se1Size ÷ 16 => se1Size, sigmoid),
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
      (channelWeights, inputTensor) -> inputTensor .* reshape(view(channelWeights, :, :), (1, 1, gf_dim * 4, :))
    ),
  ))
  return Chain(ntuple(i -> SkipConnection(seRes, +), sizeChain))
end

# U-SE-ResNet generator from original article
# https://arxiv.org/abs/2003.04685
function U_SE_ResNetGenerator(; sizeChain = 32)
  seResNetblocks = SE_ResNetChain(sizeChain)
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
    Conv((8, 8), 1 => 1; pad = (0, 1)),
    Conv((8, 8), 1 => 1),
    sigmoid,
  ) |> gpu
end

#= ConvNeXt-inspired generator
original code (pytorch): https://github.com/facebookresearch/ConvNeXt
paper: https://arxiv.org/abs/2201.03545
Metalhead.jl implementation https://github.com/FluxML/Metalhead.jl/blob/cc486bf00c60874de426dece97956528ce406564/src/convnets/convnext.jl
=#
function convNextModel(blockChannel::Int, blockRepeat::Array{Int}, maxDropPathChance::AbstractFloat)
  dropPathProb = Float32.(range(0, maxDropPathChance, sum(blockRepeat)))
  block = 0
  step = Chain[]
  channelSequence = [blockChannel * 2 ^ (stageIndex - 1) for stageIndex in axes(blockRepeat, 1)]
  for (index, rep) in enumerate(blockRepeat)
    push!( # current stage's downsample
      step,
      if block == 0
        Chain(
          Conv((4, 4), 3 => channelSequence[index]; stride = 4),
          # x -> Flux.normalise(x; dims = ndims(x) - 1),
          ChannelLayerNorm(channelSequence[index]),
        )
      else
        Chain(
          ChannelLayerNorm(channelSequence[index - 1]),
          # x -> Flux.normalise(x; dims = ndims(x) - 1),
          Conv((2, 2), channelSequence[index - 1] => channelSequence[index]; stride = 2)
        )
      end
    )
    for _ in 1:rep # current stage's ConvNeXt block stack
      block += 1
      push!(step, Chain(
        SkipConnection(
          Chain(DepthwiseConv((7, 7), channelSequence[index] => channelSequence[index]; pad = 3),
            x -> permutedims(x, (3, 1, 2, 4)),
            LayerNorm(channelSequence[index]; ϵ = 1.0f-6),
            Dense(channelSequence[index], 4 * channelSequence[index], relu),
            Dense(4 * channelSequence[index], channelSequence[index]),
            Flux.Scale(fill(Float32(1.0f-6), channelSequence[index]), false),
            x -> permutedims(x, (2, 3, 1, 4)),
            dropPathProb[block] > 0 ? stochasticDepth(dropPathProb[block]) : identity),
        +)
      ))
    end
  end
  resizeBlock = Chain(
    ConvTranspose((9, 9), channelSequence[end] => channelSequence[end] ÷ 8, stride = 6),
    ConvTranspose((9, 9), channelSequence[end] ÷ 8 => 1, stride = 6),
    Conv((9, 9), 1 => 1; pad = (3, 0)),
    Conv((9, 9), 1 => 1; pad = (3, 0)),
    Conv((10, 10), 1 => 1; pad = (3, 0)),
    sigmoid
  )
  return Chain(step..., resizeBlock) |> gpu
end