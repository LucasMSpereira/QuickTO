include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")
CUDA.allowscalar(false)
function topoGANloss(predTopo, targetTopo)
  # predTopo: batch of outputs from model
  # targetTopo: batch of respective target topologies
end
batchSize = 64
standardSize = (51, 141, 1, batchSize)
targetTopo = zeros(Float32, standardSize)
[targetTopo[1:50, 1:140, 1, sample] .= rand(Float32, (50, 140, 1)) for sample in axes(targetTopo)[4]]
# FEA data
vf = ones(Float32, standardSize) .* reshape(view(rand(Float32, batchSize), :), (1, 1, 1, :)) # VF
support = rand(Bool, standardSize) # binary highlighting pinned NODES
# components and position of loads
Fx = zeros(Float32, standardSize); Fx[10, 30, 1, :] .= 5f1; Fx[20, 40, 1, :] .= -3f1
Fy = zeros(Float32, standardSize); Fy[10, 30, 1, :] .= 1f1; Fy[20, 40, 1, :] .= 2f1
# conditioning
vm = rand(Float32, standardSize); energy = rand(Float32, standardSize)
# NN inputs
real_A = cat(vf, vm, energy; dims = 3) # generator input + conditioning
real_B = targetTopo # true topology
# disc input + conditioning + TRUE topology
real_AB = cat(vf, support, Fx, Fy, vm, energy, real_B; dims = 3)
# instantiate NNs
generator = U_SE_ResNetGenerator(); discriminator = topologyGANdisc()
# use NNs
fake_B = real_A |> gpu |> generator |> cpu # fake_B = predicted topology
fake_AB = cat(vf, support, Fx, Fy, vm, energy, fake_B; dims = 3) # disc input + conditioning + FAKE topology
Dlogits = real_AB |> gpu |> discriminator |> cpu # disc output with TRUE topology
Dlogits_ = fake_AB |> gpu |> discriminator |> cpu # disc output with FAKE topology
# Errors
vfAE = abs.(volFrac(fake_B) - volFrac(real_B)) |> mean # VF absolute error
msError = mean((fake_B .- real_B) .^ 2) # mean squared topology error
# Intermediate losses
# generator wants discriminator to output 1 to FAKE topologies
genLoss = Flux.Losses.logitbinarycrossentropy(Dlogits_, ones(Float32, batchSize)) # gan_loss_g
# discriminator should output 1 to TRUE topologies
gLossDreal = Flux.Losses.logitbinarycrossentropy(Dlogits, ones(Float32, batchSize)) # gan_loss_d_real
# discriminator should output 0 to FAKE topologies
gLossDfake = Flux.Losses.logitbinarycrossentropy(Dlogits_, zeros(Float32, batchSize)) # gan_loss_d_fake
# Final losses
genLossFinal = genLoss + l1λ * msError + l2λ * vfAE # g_loss_final. used to train generator
gLossD = gLossDreal + gLossDfake # gan_loss_d. used to train discriminator