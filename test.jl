# Packages
ENV["JULIA_CUDA_MEMORY_POOL"] = "none" # avoid GPU OOM issues
using Suppressor
@suppress_err begin
  using Random, CUDA, Statistics
  using MLUtils, Flux
  using Zygote, Optimisers, ChainRulesCore
end
const batchSize = 64
padGenOutCol = zeros(Float32, (50, 1, 1, batchSize))
padGenOutLine = zeros(Float32, (1, 141, 1, batchSize))
# import Nonconvex
# Nonconvex.@load NLopt
CUDA.allowscalar(false)
# reference number of channels used in TopologyGAN
const gf_dim = 128
const df_dim = 128
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
function padGen(genOut::Array{Float32, 4})::Array{Float32, 4}
    return cat(
        cat(genOut, padGenOutCol; dims = 2),
        padGenOutLine;
        dims = 1
    )
end
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

    # custom Flux.jl split layer
    struct Split{T}
    paths::T
    end
    Split(paths...) = Split(paths)
    Flux.@functor Split
    (m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)
#
g = U_SE_ResNetGenerator()
a = rand(Float32, (51, 141, 3, 64)) |> gpu
b = g(a)
b |> cpu |> padGen
withgradient(
    g -> mean(g(a) |> cpu |> padGen), g
)