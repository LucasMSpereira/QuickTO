# functions used to combine (mostly hdf5) files

# combine files with data for FEAloss learning pipeline
function combineFEAlossData()
  fileList = readdir(datasetPath*"data/stressCNNdata/fea loss data"; join = true, sort = false)
  globalVF = zeros(1); globalDisp = zeros(51, 141, 1)
  globalSup = zeros(3, 3, 1); globalForce = zeros(2, 4, 1)
  for file in fileList
    # open file, read data and concatenate to global arrays
    id = h5open(file, "r")
    globalVF = vcat(globalVF, read(id["vf"]))
    globalDisp = cat(globalDisp, read(id["disp"]); dims = 3)
    globalForce = cat(globalForce, read(id["force"]); dims = 3)
    globalSup = cat(globalSup, read(id["sup"]); dims = 3)
    close(id)
    @show file
  end
  # discard null initial data
  globalVF = globalVF[2:end]; globalDisp = globalDisp[:, :, 2:end]
  globalSup = globalSup[:, :, 2:end]; globalForce = globalForce[:, :, 2:end]
  count = length(globalVF)
  # create new file
  h5open(datasetPath*"data/stressCNNdata/fea loss data/FEAlossData", "w") do new
    # initialize fields in new file
    create_dataset(new, "vf", zeros(count))
    create_dataset(new, "disp", zeros(51, 141, 2*count))
    create_dataset(new, "sup", zeros(3, 3, count))
    create_dataset(new, "force", zeros(2, 4, count))
    # fill new file with data
    for i in 1:count
      new["vf"][i] = globalVF[i]
      new["sup"][:, :, i] = globalSup[:, :, i]
      new["disp"][:, :, 2*i-1 : 2*i] = globalDisp[:, :, 2*i-1 : 2*i]
      new["force"][:, :, i] = globalForce[:, :, i]
    end
  end
end

# template to combine multiple dataset hdf5 files into a new one
function combineFiles(pathRef)
  # get list of intermediate hdf5 analysis files
  files = glob("*", datasetPath*"data/analysis")
  count = 0 # global sample counter
  globalDS = []; globalSec = []; globalSID = []; globalRes = []
  for file in keys(files) # loop in files
    # open current file and read data
    id = h5open(files[file], "r")
    ds = read(id["dataset"])
    sec = read(id["section"])
    sample = read(id["sampleID"])
    res = read(id["result"])
    close(id) # close current file
    IDs = 1:length(ds) # get IDs of samples of interest in current file (criterion changes with context)
    quants = length(IDs) # number of samples of interest in current file
    globalDS = vcat(globalDS, ds[IDs]) # dataset ID of samples
    globalSec = vcat(globalSec, sec[IDs]) # section ID of samples
    globalSID = vcat(globalSID, sample[IDs]) # IDs of samples
    globalRes = vcat(globalRes, res[IDs]) # results of samples
    count += quants # update global counter
    @show count
  end
  new = h5open(datasetPath*"analyses/"*pathRef, "w") # create new file to store everything
  # initialize fields in new file
  create_dataset(new, "dataset", zeros(Int, count))
  create_dataset(new, "section", zeros(Int, count))
  create_dataset(new, "sampleID", zeros(Int, count))
  create_dataset(new, "result", zeros(count))
  # fill new file with data of interest
  for gg in 1:count
    new["dataset"][gg] = globalDS[gg] # dataset ID
    new["section"][gg] = globalSec[gg] # section ID
    new["sampleID"][gg] = globalSID[gg] # sample
    new["result"][gg] = globalRes[gg] # result value of sample
  end
  close(new) # close new file
end

# Combine pdf files into one
function combinePDFs(path, finalName; leavePDFout = " ")
  PDFfiles = filter(x -> endswith(x, "pdf"), readdir(path; join = true))
  # if PDFs with 'leavePDFout' in their name shouldn't be united
  if leavePDFout != " " 
    # amount of PDFs left out of unification
    amountOut = filter(o -> occursin(leavePDFout, o), PDFfiles) |> length
    finalName *= " " * string(amountOut + 1)
    filter!(o -> !occursin(leavePDFout, o), PDFfiles)
  end
  read(`$(Poppler_jll.pdfunite()) $(PDFfiles) $(path)/$finalName.pdf`, String) # join pdfs together
  rm.(PDFfiles)
end

# combine files for stressCNN dataset
function combineStressCNNdata(path)
  files = glob("*", path)
  count = 0 # global sample counter
  # initialize "global" variables
  globalF = zeros(2, 4, 1); globalPrin = zeros(50, 140, 1); globalVM = zeros(50, 140, 1)
  for file in keys(files) # loop in files
    id = h5open(files[file], "r") # open current file and read data
    force = read(id["forces"]); prin = read(id["principals"]); vm = read(id["vm"])
    close(id) # close current file
    quants = size(force, 3) # amount of new samples
    # concatenate file data with "global" arrays
    globalF = cat(globalF, force; dims = 3)
    globalPrin = cat(globalPrin, prin; dims = 3)
    globalVM = cat(globalVM, vm; dims = 3)
    count += quants # update global counter
    @show count
  end
  # discard null initial data
  globalF = globalF[:, :, 2:end]; globalPrin = globalPrin[:, :, 2:end]; globalVM = globalVM[:, :, 2:end]
  # create new file to store everything
  new = h5open(datasetPath*"data/stressCNNdata/stressCNNdata", "w")
  # initialize fields in new file
  create_dataset(new, "forces", zeros(2, 4, count))
  create_dataset(new, "principals", zeros(50, 140, 2*count))
  create_dataset(new, "vm", zeros(50, 140, count))
  # fill new file with data
  for sample in 1:count
    new["forces"][:, :, sample] = globalF[:, :, sample]
    new["principals"][:, :, 2*sample-1 : 2*sample] = globalPrin[:, :, 2*sample-1 : 2*sample]
    new["vm"][:, :, sample] = globalVM[:, :, sample]
  end
  close(new) # close and save new file
end