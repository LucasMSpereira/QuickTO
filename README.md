# QuickTO

Use generative adversarial networks (GANs) to accelerate 2D linear topology optimization (TO).

## Topology optimization (TO)

If a mechanical structure is under load, the optimal material distribution can be determined through iterative optimization techniques. The goal is usually to maximize the structure's natural frequency, minimize weight or maximum stress. The [TopOpt.jl](https://github.com/JuliaTopOpt/TopOpt.jl) package was used for topology optimization (dataset generation, and many related utilities).

## GANs

Generative adversarial networks uses two neural networks that compete against each other. The generator tries two create fake examples, and the discriminator tries to tell fake from real. After successful training, the generator creates convincing examples, and the discriminator is able to distinguish fake and real examples. The [Flux.jl](https://github.com/FluxML/Flux.jl) package was used for machine learning.

## GANs + TO

In the context of topology optimization, the trained generator is able to predict the optimal material distribution based on the volume fraction (amount of material), and stress and energy distributions. This prediction is immediate, avoiding the iterations in the usual optimization process. An example is shown below. The package [Makie.jl](https://github.com/MakieOrg/Makie.jl) was used for plotting.

<ul>
  <li>Top left: von Mises stress distribution in initial uniform domain.</li>
  <li>Top right: loads (arrows) and mechanical supports (dots).</li>
  <li>Bottom left: optimal design by standard SIMP topology optimization.</li>
  <li>Bottom right: optimal design suggested by trained generator.</li>
</ul>

![image](https://github.com/LucasMSpereira/QuickTO/assets/84910559/28216ee1-bc4c-428e-93bf-e8621b739a7c)

## Context

This project was developed using the [Julia](https://julialang.org) programming language. This work is the implementation behind my master's degree in mechanical engineering (2021-2023) at the Polytechnic School of University of São Paulo, Brazil.
