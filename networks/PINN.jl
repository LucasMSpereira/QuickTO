#= Initial testing of PINN implementation using NeuralPDE.jl.
After creating PINN for FEA, compare against FEA from TopOpt.jl
(in both accuracy and time). Out of these two, the best method
will be used in new training pipeline. Initial reference:
https://neuralpde.sciml.ai/stable/tutorials/pdesystem/ =#

using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
# 2D PDE
# eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)
eq = 
# Boundary conditions
bcs = [u(0,y) ~ 0.0, u(1,y) ~ 0.0,
       u(x,0) ~ 0.0, u(x,1) ~ 0.0]
# Space and time domains
domains = [x ∈ Interval(0.0,1.0),
           y ∈ Interval(0.0,1.0)]

# Neural network
dim = 2 # number of dimensions
chain = Lux.Chain(Dense(dim,16,Lux.σ),Dense(16,16,Lux.σ),Dense(16,1))

# Discretization
dx = 0.05
discretization = PhysicsInformedNN(chain,GridTraining(dx))

@named pde_system = PDESystem(eq,bcs,domains,[x,y],[u(x, y)])
prob = discretize(pde_system,discretization)

#Optimizer
opt = OptimizationOptimJL.BFGS()

#Callback function
callback = function (p,l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, opt, callback = callback, maxiters=1000)
phi = discretization.phi

using Plots

xs,ys = [infimum(d.domain):dx/10:supremum(d.domain) for d in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

u_predict = reshape([first(phi([x,y],res.u)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)