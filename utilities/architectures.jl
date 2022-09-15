# Builders for different NN architectures

# stressCNN structure 1. Predict positions and components of loads from VM field
function convMaxPoolBN(kernel, activ, ch)
  # kernel size of 1-5
  module1 = Chain(
    BatchNorm(1),
    Conv(kernel, 1 => ch, activ), # outputsize logic: 1 + (L - k)
    MaxPool(kernel), # floor(L/k)
    BatchNorm(ch),
    Conv(kernel, ch => 2*ch, activ),
    MaxPool(kernel),
    BatchNorm(2*ch),
    flatten)
  m1size = prod(Flux.outputsize(module1, (50, 140, 1, 1)))
  module2 = Chain(Dense(m1size => m1size÷2), Dense(m1size÷2 => 8))
  return Chain(module1, module2) |> gpu
end

# stressCNN structure 2. Predict positions and components of loads from VM field
function convMaxPoolBN2(kernel, activ, ch)
  # kernel size of 1-5
  module1 = Chain(
    BatchNorm(1),
    Conv(kernel, 1 => ch, activ), # outputsize logic: 1 + (L - k)
    MaxPool(kernel), # floor(L/k)
    BatchNorm(ch),
    Conv(kernel, ch => 2*ch, activ),
    MaxPool(kernel),
    BatchNorm(2*ch),
    flatten)
  m1size = prod(Flux.outputsize(module1, (50, 140, 1, 1)))
  module2 = Chain(
    Dense(m1size => Int(m1size*0.8)), Dense(Int(m1size*0.8) => Int(m1size*0.5)),
    Dense(Int(m1size*0.5) => Int(m1size*0.1)), Dense(Int(m1size*0.1) => 8)
  )
  return Chain(module1, module2) |> gpu
end

# stressCNN structure 3. Predict positions and components of loads from VM field
function convBNmeanPool(kernel, activ, ch; initialW = Flux.glorot_normal)
  module1 = Chain(
    Conv(kernel, 1 => ch, activ; init = initialW), # outputsize logic: 1 + (L - k)
    BatchNorm(ch),
    MeanPool(kernel), # floor(L/k)
    Conv(kernel, ch => 2*ch, activ; init = initialW),
    MeanPool(kernel), # floor(L/k)
    Conv(kernel, 2*ch => 4*ch, activ; init = initialW),
    flatten)
  m1size = prod(Flux.outputsize(module1, (50, 140, 1, 1)))
  module2 = Chain(Dense(m1size => m1size÷10; init = initialW), Dense(m1size÷10 => 8; init = initialW))
  return Chain(module1, module2) |> gpu
end

# stressCNN structure 4. Predict positions and components of loads from VM field
function convBNmeanPoolDrop(kernel, activ, ch; initialW = Flux.glorot_normal)
  module1 = Chain(
    Conv(kernel, 1 => ch, activ; init = initialW), # outputsize logic: 1 + (L - k)
    BatchNorm(ch),
    MeanPool(kernel), # floor(L/k)
    Conv(kernel, ch => 2*ch, activ; init = initialW),
    MeanPool(kernel), # floor(L/k)
    Conv(kernel, 2*ch => 4*ch, activ; init = initialW),
    flatten)
  m1size = prod(Flux.outputsize(module1, (50, 140, 1, 1)))
  module2 = Chain(Dropout(0.5), Dense(m1size => m1size÷10; init = initialW), Dense(m1size÷10 => 8; init = initialW))
  return Chain(module1, module2) |> gpu
end

# stressCNN structure 5. Predict positions and components of loads from VM field
function multiConvs(kernel, activ, ch)
  module1 = Chain(
    Conv(kernel, 1 => ch, activ; pad = SamePad()),
    Conv(kernel, ch => ch, activ),
    Conv(kernel, ch => 2*ch, activ),
    Conv(kernel, 2*ch => 2*ch, activ),
    Conv(kernel, 2*ch => 4*ch, activ),
    Conv(kernel, 4*ch => 4*ch, activ),
    flatten)
  m1size = prod(Flux.outputsize(module1, (50, 140, 1, 1)))
  module2 = Chain(Dense(m1size => 8))
  return Chain(module1, module2) |> gpu
end

# custom split layer
struct Split{T}
  paths::T
end
Split(paths...) = Split(paths)
Flux.@functor Split
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

# stressCNN structure 5. Predict positions and components of loads from VM field
function multiOutputs(kernel, activ, ch)
  module1 = Chain(
    Conv(kernel, 2 => ch, activ; pad = SamePad()),
    Conv(kernel, ch => ch, activ),
    Conv(kernel, ch => 2*ch, activ),
    Conv(kernel.÷2, 2*ch => 2*ch, activ),
    Conv(kernel.÷2, 2*ch => 4*ch, activ),
    Conv(kernel.÷2, 4*ch => 4*ch, activ),
    flatten)
  m1size = prod(Flux.outputsize(module1, (51, 141, 2, 1)))
  module2 = Split(Dense(m1size => 2), Dense(m1size => 2), Dense(m1size => 2), Dense(m1size => 2))
  return Chain(module1, module2) |> gpu
end