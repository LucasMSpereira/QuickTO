using Base.Iterators: partition
using Printf
using Statistics
using Random
using Images
using Parameters: @with_kw
using Flux
using Flux.Data: DataLoader
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using MLDatasets: MNIST
using CUDA

@with_kw struct HyperParams
  batch_size::Int = 128
  latent_dim::Int = 100
  epochs::Int = 25
  verbose_freq::Int = 1000
  output_dim::Int = 5
  disc_lr::Float64 = 0.0002
  gen_lr::Float64 = 0.0002
  device::Function = gpu
end

# Function used to load the MNIST images
function load_MNIST_images(hparams)
  images = MNIST.traintensor(Float32)

  # Normalize the images to (-1, 1)
  normalized_images = @. 2f0 * images - 1f0
  image_tensor = reshape(normalized_images, 28, 28, 1, :)

  # Create a dataloader that iterates over mini-batches of the image tensor
  dataloader = DataLoader(image_tensor, batchsize=hparams.batch_size, shuffle=true)

  return dataloader
end

dalo = load_MNIST_images(HyperParams)

# Function for intializing the model weights with values 
# sampled from a Gaussian distribution with μ=0 and σ=0.02
dcgan_init(shape...) = randn(Float32, shape) * 0.02f0

# Define architecture of generator
function Generator(latent_dim)
  Chain(
      Dense(latent_dim, 7*7*256, bias=false),
      BatchNorm(7*7*256, relu),

      x -> reshape(x, 7, 7, 256, :),

      ConvTranspose((5, 5), 256 => 128; stride = 1, pad = 2, init = dcgan_init, bias=false),
      BatchNorm(128, relu),

      ConvTranspose((4, 4), 128 => 64; stride = 2, pad = 1, init = dcgan_init, bias=false),
      BatchNorm(64, relu),

      # The tanh activation ensures that output is in range of (-1, 1)
      ConvTranspose((4, 4), 64 => 1, tanh; stride = 2, pad = 1, init = dcgan_init, bias=false),
  )
end

# Create a dummy generator of latent dim 100
generator = Generator(100)
noise = randn(Float32, 100, 3) # The last axis is the batch size

# Feed the random noise to the generator
gen_image = generator(noise)
@assert size(gen_image) == (28, 28, 1, 3)

# Definition of discriminator
function Discriminator()
  Chain(
      Conv((4, 4), 1 => 64; stride = 2, pad = 1, init = dcgan_init),
      x->leakyrelu.(x, 0.2f0),
      Dropout(0.3),

      Conv((4, 4), 64 => 128; stride = 2, pad = 1, init = dcgan_init),
      x->leakyrelu.(x, 0.2f0),
      Dropout(0.3),

      # The output is now of the shape (7, 7, 128, batch_size)
      flatten,
      Dense(7 * 7 * 128, 1) 
  )
end

# Dummy Discriminator
discriminator = Discriminator()
# We pass the generated image to the discriminator
logits = discriminator(gen_image)
@assert size(logits) == (1, 3)

function discriminator_loss(real_output, fake_output)
  real_loss = logitbinarycrossentropy(real_output, 1)
  fake_loss = logitbinarycrossentropy(fake_output, 0)
  return real_loss + fake_loss
end

generator_loss(fake_output) = logitbinarycrossentropy(fake_output, 1)