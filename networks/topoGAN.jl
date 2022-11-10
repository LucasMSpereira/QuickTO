include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
CUDA.allowscalar(false)
function topoGANloss(predTopo, targetTopo)
  # predTopo: batch of outputs from model
  # targetTopo: batch of respective target topologies
end
batchSize = 64
standardSize = (51, 141, 1, batchSize)
# Vectors filled with 1.0 and 0.0, respectively. To be
# used in binary cross-entropy with logits
const discTrue = ones(Float32, batchSize)
const discFake = zeros(Float32, batchSize)
targetTopo = pad_constant(
  rand(Float32, (50, 140, 1, batchSize)),
  (0, 1, 0, 1); dims = [1 2]
)
# FEA data
vf = ones(Float32, standardSize) .* reshape(view(rand(Float32, batchSize), :), (1, 1, 1, :)) # VF
support = rand(Bool, standardSize) # binary highlighting pinned NODES
# components and position of loads
Fx = zeros(Float32, standardSize); Fx[10, 30, 1, :] .= 5f1; Fx[20, 40, 1, :] .= -3f1
Fy = zeros(Float32, standardSize); Fy[10, 30, 1, :] .= 1f1; Fy[20, 40, 1, :] .= 2f1
# conditioning
vm = rand(Float32, standardSize); energy = rand(Float32, standardSize)
# NN inputs
genInput = cat(vf, vm, energy; dims = 3) # generator input + conditioning
# disc input + conditioning + TRUE topology
discInputReal = cat(genInput, support, Fx, Fy, targetTopo; dims = 3)
# instantiate NNs
generator = U_SE_ResNetGenerator(); discriminator = topologyGANdisc()
# Get outputs from generator and discriminator
function evaluations(genInput_, discInputAndConds, targetTopo_)
  # genInput_: batch of generator inputs
  # discInputAndConds: batch of incomplete inputs for discriminator
  # targetTopo_: batch of target topologies
  # batch of predicted (FAKE) topologies
  predTopo_ = genInput_ |> gpu |> generator |> cpu
  return (predTopo_,
    # disc input + conditioning + FAKE topology
    cat(discInputAndConds..., predTopo_; dims = 3) |> gpu |> discriminator |> cpu,
    # disc input + conditioning + TRUE topology
    cat(discInputAndConds..., targetTopo_; dims = 3) |> gpu |> discriminator |> cpu
  )
end
# batch of outputs from generator and discriminator
predTopo, logitFake, logitReal = evaluations(genInput, (genInput, support, Fx, Fy), targetTopo)
# Errors used for loss calculation in topologyGAN
function topoGANerrors(predTopo_, targetTopo_)
  # predTopo_: batch of predicted (FAKE) topologies
  # targetTopo_: batch of target topologies
  return (
    # mean of batch volume fraction absolute error
    abs.(volFrac(predTopo_) .- volFrac(targetTopo_)) |> mean,
    # batch mean squared topology error
    (predTopo_ .- targetTopo_) .^ 2 |> mean
    )
end
# Errors used for loss calculation
vfAE, msError = topoGANerrors(predTopo, targetTopo)
# logits representing the discriminator's behavior
function topoGANlogits(logitFake_, logitReal_)
  # discTrue: vector filled with 1.0; discFake: vector filled with 0.0
  return (
    # generator wants discriminator to output 1 to FAKE topologies
    Flux.Losses.logitbinarycrossentropy(logitFake_, discTrue), # gan_loss_g
    # discriminator should output 0 to FAKE topologies
    Flux.Losses.logitbinarycrossentropy(logitFake_, discFake), # gan_loss_d_fake
    # discriminator should output 1 to TRUE topologies
    Flux.Losses.logitbinarycrossentropy(logitReal_, discTrue) # gan_loss_d_real
  )
end
# logits representing the discriminator's behavior
genLossLogit, gLossDfake, gLossDreal = topoGANlogits(logitFake, logitReal)
# Final losses
function finalTopoGANlosses(genLossLogit_, msError_, vfAE_, gLossDreal_, gLossDfake_)
  return (
    genLossLogit_ + l1λ * msError_ + l2λ * vfAE_, # g_loss_final. used to train generator
    gLossDreal_ + gLossDfake_ # gan_loss_d. used to train discriminator
  )
end
# Generator and discriminator final losses for batch of inputs
genLossFinal, gLossD = finalTopoGANlosses(genLossLogit, msError, vfAE, gLossDreal, gLossDfake)