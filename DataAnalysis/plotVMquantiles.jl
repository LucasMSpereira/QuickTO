# Script to find and plot median and outlier samples based on max VM

# function definitions
include("C:/Users/LucasKaoid/Meu Drive/Estudo/Poli/Pesquisa/Programas/QuickTO/QuickTO/QTOutils.jl")

# open vm file and get data
file = h5open("C:/Users/LucasKaoid/Desktop/datasets/post/plastification/vmAll", "r")
ds = read(file["dataset"])
vmAll = read(file["vm"])
sID = read(file["sampleID"])
sec = read(file["section"])
close(file)

### locate samples
# sortVM = sort(vmAll; rev = true) # sort from greatest to lowest
## median sample
# medianSample = sortVM[round(Int, length(sortVM)/2)] # median value of max vm
# medianSampleID = findfirst(x -> x == medianSample, vmAll) # position of median vm in file
medianSampleID = sortperm(vmAll)[round(Int, length(vmAll)/2)] # position of median vm in file
medianSample = vmAll[medianSampleID] # median value of max vm
medianSampleDS = ds[medianSampleID] # dataset ID of median sample
medianSample_sID = sID[medianSampleID] # sample ID of median sample
medianSampleSEC = sec[medianSampleID] # section ID of median sample
## outlier samples
# quartiles
q75 = quantile(vmAll, 0.75) # 0.4023
q25 = quantile(vmAll, 0.25) # 0.2662
# interquartile range
IQR = q75 - q25 # 0.1361 * 1.5 = 0.2042
# indices of outliers
outlierBoundary = 17*IQR + q75 # criterion for outlier
outlierSamples = findall(x -> x > outlierBoundary, vmAll) # 6758 samples (1.5*IQR)
length(outlierSamples)/length(vmAll) # 0.0488 (5% of dataset) (1.5*IQR)
outlierSampleDS = ds[outlierSamples] # dataset ID of each outlier
outlierSample_sID = sID[outlierSamples] # sample ID of each outlier
outlierSampleSEC = sec[outlierSamples] # section ID of each outlier

wantedSamples = hcat(
  vcat(medianSampleDS, outlierSampleDS),
  vcat(medianSampleSEC, outlierSampleSEC),
  vcat(medianSample_sID, outlierSample_sID),
)

# build toy struct
@with_kw mutable struct FEAparameters
  quants::Int = 1 # number of TO problems per section
  V::Array{Real} = [0.4+rand()*0.5 for i in 1:quants] # volume fractions
  problems::Any = Array{Any}(undef, quants) # store FEA problem structs
  meshSize::Tuple{Int, Int} = (140, 50) # Size of rectangular mesh
  elementIDarray::Array{Int} = [i for i in 1:prod(meshSize)] # Vector that lists element IDs
  # matrix with element IDs in their respective position in the mesh
  elementIDmatrix::Array{Int,2} = convert.(Int, quad(meshSize...,[i for i in 1:prod(meshSize)]))
  section::Int = 1 # Number of dataset HDF5 files with "quants" samples each
end
FEAparams = FEAparameters(); FEAparams = problem!(FEAparams)

function vmPlots(id)
  files = glob("*", "C:/Users/LucasKaoid/Desktop/datasets/data/$id") # get list of file names in folder "id"
  count = 0 # global counter of samples
  colSize = 500
  for file in keys(files) # loop in files
    time = @elapsed begin
      currentDS, currentSec = getIDs(files[file]) # get dataset and section IDs of current file
      for sample in 1:size(wantedSamples, 1) # check which of the wanted samples the current file contains, if any
        if currentDS == wantedSamples[sample, 1] && currentSec == wantedSamples[sample, 2]
          i = Int(wantedSamples[sample, 3])
          force, supps, vf, disp, topo = getDataFSVDT(files[file]) # get data from current file
          # get vm field
          vm = calcVM(
            prod(FEAparams.meshSize), FEAparams,
            disp[:,:, (2 * i - 1) : (2 * i)],
            210e3 * vf[i], 0.3
          )
          # create makie figure and set it up
          fig = Figure(resolution = (1400, 700));
          colsize!(fig.layout, 1, Fixed(colSize))
          # labels for first line of grid
          Label(fig[1, 1], "Supports", textsize = 20)
          Label(fig[1, 2], "Force positions", textsize = 20)
          colsize!(fig.layout, 2, Fixed(colSize))
          supports = zeros(FEAparams.meshSize)'
          if supps[:,:,i][1,3] > 3


            if supps[:,:,i][1,3] == 4
              # left
              supports[:,1] .= 3
            elseif supps[:,:,i][1,3] == 5
              # bottom
              supports[end,:] .= 3
            elseif supps[:,:,i][1,3] == 6
              # right
              supports[:,end] .= 3
            elseif supps[:,:,i][1,3] == 7
              # top
              supports[1,:] .= 3
            end


          else

            [supports[supps[:,:,i][m, 1], supps[:,:,i][m, 2]] = 3 for m in 1:size(supps[:,:,i])[1]]

          end
          # plot supports
          heatmap(fig[2,1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,supports')
          # plot forces
          quantForces = size(force[:,:,1], 1)
          loadXcoord = zeros(quantForces)
          loadYcoord = zeros(quantForces)
          for l in 1:quantForces
            loadXcoord[l] = force[:,:,i][l,2]
            loadYcoord[l] = FEAparams.meshSize[2] - force[:,:,i][l,1] + 1
          end
          axis = Axis(fig[2,2])
          xlims!(axis, -round(0.1*FEAparams.meshSize[1]), round(1.1*FEAparams.meshSize[1]))
          ylims!(axis, -round(0.1*FEAparams.meshSize[2]), round(1.1*FEAparams.meshSize[2]))
          arrows!(
            axis, loadXcoord, loadYcoord,
            force[:, :, i][:,3], force[:, :, i][:,4];
            # norm of weakest force. will be used to scale force vectors in arrows!() command
            lengthscale = 1 / (0.1*minimum( sqrt.( (force[:, :, i][:,3]).^2 + (force[:, :, i][:,4]).^2 ) ))
          )
          # text with values of force components
          f1 = "Forces (N):\n1: $(round(Int,force[:,:,i][1, 3])); $(round(Int,force[:,:,i][1, 4]))\n"
          if sample == 1
            f2 = "2: $(round(Int,force[:,:,i][2, 3])); $(round(Int,force[:,:,i][2, 4]))\nMedian"
          else
            f2 = "2: $(round(Int,force[:,:,i][2, 3])); $(round(Int,force[:,:,i][2, 4]))"
          end
          # sampleQuantile = "quantile: $( round((findfirst(x -> isapprox(x, maximum(vm); atol = 1e-21), sortVM ) - 1) / length(sortVM) * 100; digits = 1 ) )%"
          l = text(
              fig[2,3],
              f1*f2;
          )

          #
              l.axis.attributes.xgridvisible = false
              l.axis.attributes.ygridvisible = false
              l.axis.attributes.rightspinevisible = false
              l.axis.attributes.leftspinevisible = false
              l.axis.attributes.topspinevisible = false
              l.axis.attributes.bottomspinevisible = false
              l.axis.attributes.xticksvisible = false
              l.axis.attributes.yticksvisible = false
              l.axis.attributes.xlabelvisible = false
              l.axis.attributes.ylabelvisible = false
              l.axis.attributes.titlevisible  = false
              l.axis.attributes.xticklabelsvisible = false
              l.axis.attributes.yticklabelsvisible = false
              l.axis.attributes.tellheight = false
              l.axis.attributes.tellwidth = false
              l.axis.attributes.halign = :center
              l.axis.attributes.width = 300
          #
          
          # labels for second line of grid
          Label(fig[3, 1], "Topology VF = $(round(vf[i];digits=3))", textsize = 20)
          Label(fig[3, 2], "von Mises (MPa)", textsize = 20)
          # plot final topology
          heatmap(fig[4, 1],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,topo[:,:,i]')
          # plot von Mises
          _,hm = heatmap(fig[4, 2],1:FEAparams.meshSize[2],FEAparams.meshSize[1]:-1:1,vm')
          # setup colorbar for von Mises
          bigVal = ceil(maximum(vm)) + 1
          t = floor(0.2*bigVal)
          t == 0 && (t = 1)
          Colorbar(fig[4, 3], hm, ticks = 0:t:bigVal)
          # save image file
          standardPath = "C:/Users/LucasKaoid/Desktop/datasets/post/plastification/quantilePics/"
          save(standardPath*"$(wantedSamples[sample, 1]) $(wantedSamples[sample, 2]) $(wantedSamples[sample, 3]).png", fig)
          count += 1
          println("image $count \t $( round(Int, count / size(wantedSamples, 1) * 100) )%")
        end
      end
    end
  end
end
@time [@time vmPlots(g) for g in 1:6]