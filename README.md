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

## Using the project

The easiest way to use and experiment with the code is through the notebook QuickTO.ipynb in google colab. However, connection stability, GPU acccess and disk space constraints prevent experiments without **pro** google colab.

For local reproduction, follow these steps:

<ul>
  <li>Download and install Julia 1.9.0 (or higher): https://julialang.org/downloads/.</li>
  <li>Follow the necessary procedures to install the CUDA.jl package: https://github.com/JuliaGPU/CUDA.jl.</li>
  <li>Clone this repository.</li>
  <li>Download and extract the dataset zip (17 GB when compressed): https://zenodo.org/record/8191138.</li>
  <li>Create the directory path networks\GANplots inside the project folder.</li>
  <li>Create folder "checkpoints" inside the dataset folder.</li>
  <li>Every new session, start by running QTOutils.jl to include packages and setup a few definitions. Do this by openning a Julia REPL (obtained in the first step) in the project's folder. Then enter the command "include("./QTOutils.jl")". In a new computer, the first run of this script will result in the REPL giving the option to install the packages.</li>
  <li>Setup main script topoGAN.jl, including dataset path and project path constants. Running this script will start the training procedures, save the final models, and save a pdf report.</li>
</ul>

To interact with the code, it is recommended to use the [VS code](https://code.visualstudio.com) IDE with the Julia extension.

## Context

This project was developed using the [Julia](https://julialang.org) programming language. This work is the implementation behind my master's degree in mechanical engineering (2021-2023) at the Polytechnic School of University of São Paulo, Brazil.
