using Statistics, Flux, MLUtils, Zygote, LinearAlgebra, Random, Optimisers
Random.seed!(3111)
const genInput_ = rand(Float32, (10, 10, 3, 5))
const condition_ = rand(Float32, (10, 10, 3, 5))
const label_ = rand(Float32, (10, 10, 1, 5))
function models()
    gen = Chain(Conv((5, 5), 3 => 1, pad = SamePad()),
        ConvTranspose((5, 5), 1 => 1, pad = SamePad()),
    )
    disc = Chain(Conv((5, 5), 7 => 1, pad = SamePad()), flatten,
        Dense(100 => 1, leakyrelu)
    )
    return gen |> gpu, disc |> gpu
end
volFrac(x) = [mean(x[:, :, :, sample]) for sample in axes(x, 4)]
reshapeDiscOut(x) = dropdims(x |> transpose |> Array; dims = 2)
function GANgradsMWE(
  gen, disc, genInput, condition, label;
  genState = Optimisers.setup(Optimisers.Adam(), gen),
  discState = Optimisers.setup(Optimisers.Adam(), disc)
)
  discOutFake, discInputFake = 0.0, 0.0 # initialize for scope purposes
  function genLoss(genOutput) # generator loss. Defined here for scope purposes
    mse = (genOutput .- label) .^ 2 |> mean
    absError = abs.(volFrac(genOutput) .- volFrac(label)) |> mean
    discInputFake = cat(genInput, condition, genOutput; dims = 3) |> gpu
    discOutFake = discInputFake |> disc |> cpu |> reshapeDiscOut
    return Flux.Losses.logitbinarycrossentropy(
      discOutFake, ones(size(discOutFake))
    ) + 10_000 * mse + 1 * absError
  end
  function discLoss(discOutReal, discOutFake) # discriminator loss
    return Flux.Losses.logitbinarycrossentropy(
      discOutReal, ones(discOutReal |> size)
    ) + Flux.Losses.logitbinarycrossentropy(
      discOutFake, zeros(discOutFake |> size)
    )
  end
  genInputGPU = genInput |> gpu
  discInputReal = cat(genInput, condition, label; dims = 3) |> gpu
  genLossVal_, genGrads_ = withgradient(
    gen -> genLoss(gen(genInputGPU) |> cpu), gen
  )
  discLossVal_, discGrads_ = withgradient(
    disc -> discLoss(
        disc(discInputReal) |> cpu |> reshapeDiscOut,
        disc(discInputFake) |> cpu |> reshapeDiscOut
    ),
    disc
  )
  genState, gen = Flux.Optimise.update!(genState, gen, genGrads_[1])
  discState, disc = Flux.Optimise.update!(discState, disc, discGrads_[1])
  return genGrads_, genLossVal_, discGrads_, discLossVal_
end
genGrads, genLossVal, discGrads, discLossVal = GANgradsMWE(
    models()..., genInput_, condition_, label_
);
LinearAlgebra.norm(::Nothing, p::Real = 2) = false
@show norm(genGrads); @show norm(discGrads);