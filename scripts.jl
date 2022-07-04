include("./QTOutils.jl")

if false
# struct with general parameters
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
FEAparams = FEAparameters()

# include toy problem in "problems" field to get mesh references
FEAparams = problem!(FEAparams)
end

# Function to be applied to every sample
function func(forces, supps, vf, disp, top, dataset, section, sample)
  #=
    product of lengths of A* paths connecting extreme elements
    in binary version of final topology. If null, this sample
    suffers from structural disconnection and a heatmap plot
    will be generated
  =#
  lengthProds = disconnections(top, dataset, section, sample)
  return lengthProds

end

println(
  "call 'processDataset(func, id, numFiles)'.
  'func' - function to be applied to every sample
  'id' - name of folder
  (optional) 'numFiles' - string with number of
    files to analyze in that folder,
    in case of partial analysis"
)

#=
concatenar 5 amostras a serem removidas por não linearidade geométrica com as amostras
de desconexões. então usar esse novo arquivo relativo às duas análises para remover amostras
do dataset. na construção desse novo arquivo, evitar repetição de amostras
=#

file = h5open("C:/Users/LucasKaoid/Desktop/datasets/data/1/1940 3 907", "r")
data = read.(HDF5.get_datasets(file))
ds = read(file["dataset"])
res = read(file["result"])
sID = read(file["sampleID"])
sec = read(file["section"])
close(file)

nullDiscID = findall(x -> x == 0, resDisc) # indices of samples with disconnections
nonLinID = findall(x -> x > 5, resNlin) # indices of samples with geom. nonlinearity
nSamples = length(nullDiscID) + length(nonLinID)
ds = cat(dsDisc[nullDiscID], dsNlin[nonLinID]; dims = 1) # datasets of samples of interest
sID = cat(sIDdisc[nullDiscID], sIDnLin[nonLinID]; dims = 1) # IDs of samples of interest
sec = cat(secDisc[nullDiscID], secNlin[nonLinID]; dims = 1) # sections of samples of interest
newRes = zeros(nSamples)
new = h5open("C:/Users/LucasKaoid/Desktop/datasets/post/nLinearEdisconnect", "w") # create new file to store everything
# initialize fields in new file
create_dataset(new, "dataset", zeros(Int, nSamples))
create_dataset(new, "section", zeros(Int, nSamples))
create_dataset(new, "sampleID", zeros(Int, nSamples))
create_dataset(new, "result", zeros(nSamples))
# fill new file with data of interest
for gg in 1:nSamples
  new["dataset"][gg] = ds[gg] # dataset ID
  new["section"][gg] = sec[gg] # section ID
  new["sampleID"][gg] = sID[gg] # sample
  new["result"][gg] = newRes[gg] # result value of sample
end
close(new) # close new file

numFolder = 0
for folder in 1:6
  files = glob("*", "C:/Users/LucasKaoid/Desktop/datasets/data/$folder")
  global numFolder += numSample(files)
end
println(numFolder)