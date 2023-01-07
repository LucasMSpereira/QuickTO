include(projPath * "QTOutils.jl")
const batchSize = 64
const normalizeDataset = true # choose to normalize data in [-1; 1]
const startTime = timeNow()
# const to = TimerOutput()
const percentageDataset = 0.008 # fraction of dataset to be used
LinearAlgebra.norm(::Nothing, p::Real=2) = false
# toyDisc = @autosize (51, 141, 7, 64) Chain(
#     Conv((5, 5), 7 => 1), flatten,
#     Dense(_ => 1)
# )
mdNew = GANmetaData(
    # Chain(Conv((5, 5), 3 => 1, pad = (0, 3, 0, 3))),
    U_SE_ResNetGenerator(),
    # topologyGANdisc(),
    Chain(BatchNorm(7), flatten) |> gpu,
    Flux.Optimise.Adam(), Flux.Optimise.Adam(),
    epochTrainConfig(1, 1)
)

function gpTerm(disc_, genInput_, FEAinfo_, realTopo, fakeTopo)
    # interpolate fake and true topologies
    interpTopos = interpolateTopologies(realTopo, fakeTopo)
    discInputInterp = solidify(genInput_, FEAinfo_, interpTopos) |> gpu
    # use disc and get gradient
    println("11")
    grads = Flux.gradient(discInputInterp -> disc_(discInputInterp) |> cpu |> reshapeDiscOut, discInputInterp)
    println("12")
    println("gpNorm: ", norm(grads[1]), "  gpTerm: ", (norm(grads[1]) - 1) ^ 2)
    return (norm(grads[1]) - 1) ^ 2
end

function wganGPepoch(metaData, goal, nCritic)
    # initialize variables related to whole epoch
    genLossHist, discLossHist, batchCount = 0.0, 0.0, 0
    groupFiles = defineGroupFiles(metaData, goal)
    gen = getGen(metaData); disc = getDisc(metaData)
    # loop in groups of files used for current split
    for (groupIndex, group) in enumerate(groupFiles)
        # get loader with data for current group
        currentLoader = GANdataLoader(
            metaData, goal, group;
            lastFileBatch = groupIndex == length(groupFiles)
        )
        # iterate in batches of data
        for (genInput, FEAinfo, realTopology) in currentLoader
            batchCount += 1
            GC.gc(); desktop && CUDA.reclaim()
            if batchCount <= nCritic # disc batch
                println("disc batch ", batchCount)
                # generate fake topologies
                fakeTopology = gen(genInput |> gpu) |> cpu |> padGen
                # discriminator inputs with real and fake topologies
                discInputReal = solidify(genInput, FEAinfo, realTopology) |> gpu
                discInputFake = solidify(genInput, FEAinfo, fakeTopology) |> gpu
                function wganGPloss(discOutReal, discOutFake)
                    return mean(discOutFake) - mean(discOutReal) + 10 * gpTerm(
                        disc, genInput, FEAinfo, realTopology, fakeTopology
                    )
                end
                if goal == :train
                    lossVal, discGrads = withgradient(
                        disc -> wganGPloss(
                            disc(discInputReal) |> cpu |> reshapeDiscOut,
                            disc(discInputFake) |> cpu |> reshapeDiscOut
                        ), disc
                    )
                    Flux.Optimise.update!(
                        metaData.discDefinition.optInfo.optState, disc, discGrads[1]
                    )
                    println("""
                    mean(discOutFalse): $(disc(discInputReal) |> cpu |> reshapeDiscOut |> mean)
                    mean(discOutFalse): $(disc(discInputFake) |> cpu |> reshapeDiscOut |> mean)
                    lossVal: $lossVal  norm(discGrads): $(norm(discGrads))
                    """)
                else
                    loss = wganGPloss(disc(discInputReal) |> cpu |> reshapeDiscOut,
                        disc(discInputFake) |> cpu |> reshapeDiscOut)
                end
            else # generator batch
                println("gen batch ", batchCount)
                batchCount = 0
                if goal == :train
                    genLossVal, genGrads = withgradient(
                        gen -> -solidify(
                            genInputGPU, FEAinfoGPU, gen(genInputGPU) |> cpu |> padGen
                        ) |> disc |> cpu |> mean, gen
                    )
                    Flux.Optimise.update!(
                        metaData.genDefinition.optInfo.optState, gen, genGrads[1]
                    )
                    @show genLossVal; @show norm(genGrads)
                    println("""
                    genLossVal: $genLossVal  norm(genGrads): $(norm(genGrads))
                    """)
                else
                    genLossVal = -solidify(
                        genInputGPU, FEAinfoGPU, gen(genInputGPU) |> cpu |> padGen
                    ) |> disc |> cpu |> mean
                end
                return nothing
            end
        end
    end
end

wganGPepoch(mdNew, :train, 5)

function mwe()
    toyDisc = @autosize (51, 141, 7, 64) Chain(
        Conv((5, 5), 7 => 3),
        BatchNorm(3),
        flatten,
        Dense(_ => 1)
    )
    toyDiscGPU = gpu(toyDisc)
    realDisc = topologyGANdisc()
    input = rand(Float32, (51, 141, 7, 64)) |> gpu
    grads = Flux.gradient(input -> toyDiscGPU(input) |> mean, input)
    println(norm(grads[1]))
    realGrads = Flux.gradient(input -> realDisc(input) |> mean, input)
    println(norm(realGrads[1]))
end
# mwe()
function gpTermMWE()
    discriminator = topologyGANdisc()
    realTopology = rand(Float32, (51, 141, 1, 64))
    fakeTopology = rand(Float32, (51, 141, 1, 64))
    genInput = rand(Float32, (51, 141, 3, 64))
    FEAinfo = rand(Float32, (51, 141, 3, 64))
    # interpolate fake and true topologies
    ϵ = reshape(rand(Float32, size(realTopology, 4)), (1, 1, 1, size(realTopology, 4)))
    interpTopos = @. ϵ * realTopology + (1 - ϵ) * fakeTopology
    discInputInterpGPU = cat(genInput, FEAinfo, interpTopos; dims = 3) |> gpu
    # use disc and get gradient
    placeHolderLoss(discOut) = mean(dropdims(discOut |> transpose |> Array; dims = 2))
    println("11")
    grads = Flux.gradient(
        discInputInterpGPU -> placeHolderLoss(discriminator(discInputInterpGPU) |> cpu),
        discInputInterpGPU
    )
    println("12")
    println("gpNorm: ", norm(grads[1]), "  gpTerm: ", (norm(grads[1]) - 1) ^ 2)
    return (norm(grads[1]) - 1) ^ 2
end
# gpTermMWE()