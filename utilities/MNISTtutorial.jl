# # Define functions related to machine learning

# # Output loss and accuracy during training (CNN MNIST tutorial)
# function eval_loss_accuracy(loader, model, device)
#   l = 0f0
#   acc = 0
#   ntot = 0
#   for (x, y) in loader
#       x, y = x |> device, y |> device
#       ŷ = model(x)
#       l += loss(ŷ, y) * size(x)[end]        
#       acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
#       ntot += size(x)[end]
#   end
#   return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
# end

# # Load MNIST data from MLDatasets (CNN Flux zoo MNIST tutorial)
# function get_data(args)
#   xtrain, ytrain = MNIST(:train)[:]
#   xtest, ytest = MNIST(:test)[:]

#   xtrain = reshape(xtrain, 28, 28, 1, :)
#   xtest = reshape(xtest, 28, 28, 1, :)

#   ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

#   train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
#   test_loader = DataLoader((xtest, ytest),  batchsize=args.batchsize)
  
#   return train_loader, test_loader
# end

# # Build NN architecture (CNN MNIST tutorial)
# function LeNet5(; imgsize=(28,28,1), nclasses=10) 
#   out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
  
#   return Chain(
#           Conv((5, 5), imgsize[end]=>6, relu),
#           MaxPool((2, 2)),
#           Conv((5, 5), 6=>16, relu),
#           MaxPool((2, 2)),
#           flatten,
#           Dense(prod(out_conv_size), 120, relu), 
#           Dense(120, 84, relu), 
#           Dense(84, nclasses)
#         )
# end

# # Train function for CNN MNIST tutorial
# function train(; kws...)
#   # * Checks whether there is a GPU available and uses it for training the model. Otherwise, it uses the CPU.
#   # * Loads the MNIST data using the function `get_data`.
#   # * Creates the model and uses the [ADAM optimiser](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.ADAM) with weight decay.
#   # * Loads the [TensorBoardLogger.jl](https://github.com/JuliaLogging/TensorBoardLogger.jl) for logging data to Tensorboard.
#   # * Creates the function `report` for computing the loss and accuracy during the training loop. It outputs these values to the TensorBoardLogger.
#   # * Runs the training loop using [Flux’s training routine](https://fluxml.ai/Flux.jl/stable/training/training/#Training). For each epoch (step), it executes the following:
#   #   * Computes the model’s predictions.
#   #   * Computes the loss.
#   #   * Updates the model’s parameters.
#   #   * Saves the model `model.bson` every `checktime` epochs (defined as argument above.)
#   args = Args(; kws...)
#   args.seed > 0 && Random.seed!(args.seed)
#   use_cuda = args.use_cuda && CUDA.functional()
  
#   if use_cuda
#       device = gpu
#       @info "Training on GPU"
#   else
#       device = cpu
#       @info "Training on CPU"
#   end

#   ## DATA
#   train_loader, test_loader = get_data(args)
#   @info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

#   ## MODEL AND OPTIMIZER
#   model = LeNet5() |> device
#   @info "LeNet5 model: $(num_params(model)) trainable params"    
  
#   ps = Flux.params(model)  

#   opt = ADAM(args.η) 
#   if args.λ > 0 ## add weight decay, equivalent to L2 regularization
#       opt = Optimiser(WeightDecay(args.λ), opt)
#   end
  
#   ## LOGGING UTILITIES
#   if args.tblogger 
#       tblogger = TBLogger(args.savepath, tb_overwrite)
#       set_step_increment!(tblogger, 0) ## 0 auto increment since we manually set_step!
#       @info "TensorBoard logging at \"$(args.savepath)\""
#   end
  
#   function report(epoch)
#       train = eval_loss_accuracy(train_loader, model, device)
#       test = eval_loss_accuracy(test_loader, model, device)        
#       println("Epoch: $epoch   Train: $(train)   Test: $(test)")
#       if args.tblogger
#           set_step!(tblogger, epoch)
#           with_logger(tblogger) do
#               @info "train" loss=train.loss  acc=train.acc
#               @info "test"  loss=test.loss   acc=test.acc
#           end
#       end
#   end
  
#   ## TRAINING
#   @info "Start Training"
#   report(0)
#   for epoch in 1:args.epochs
#       @showprogress for (x, y) in train_loader
#           x, y = x |> device, y |> device
#           gs = Flux.gradient(ps) do
#                   ŷ = model(x)
#                   loss(ŷ, y)
#               end

#           Flux.Optimise.update!(opt, ps, gs)
#       end
      
#       ## Printing and logging
#       epoch % args.infotime == 0 && report(epoch)
#       if args.checktime > 0 && epoch % args.checktime == 0
#           !ispath(args.savepath) && mkpath(args.savepath)
#           modelpath = joinpath(args.savepath, "model.bson") 
#           let model = cpu(model) ## return model to cpu before serialization
#               BSON.@save modelpath model epoch
#           end
#           @info "Model saved in \"$(modelpath)\""
#       end
#   end
# end

# round4(x) = round(x, digits=4)

# # get total amount of model parameters
# num_params(model) = sum(length, Flux.params(model)) 
