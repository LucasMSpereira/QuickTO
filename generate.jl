using TopOpt, Makie
using TopOpt.TopOptProblems.Visualization: visualize
import GLMakie

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 10.0; # downward force
n = 2

nels = (60, 20)
problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, v, f);
result = Array{Any}(undef, n)
simps = Array{Any}(undef, n)
V = rand(n)

@time for i in 1:n

    xmin = 1e-6 # minimum density
    rmin = 2.0; # density filter radius

    penalty = TopOpt.PowerPenalty(3.0)
    solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)

    comp = TopOpt.Compliance(problem, solver)
    filter = DensityFilter(solver; rmin=rmin)
    obj = x -> comp(filter(x))

    volfrac = TopOpt.Volume(problem, solver)
    constr = x -> volfrac(filter(x)) - V[i]

    mma_options =
        options = MMAOptions(;
            maxiter=3000, tol=Nonconvex.Tolerance(; x=1e-3, f=1e-3, kkt=0.001)
        )
    convcriteria = Nonconvex.KKTCriteria()
    x0 = fill(V[i], length(solver.vars))
    optimizer = Optimizer(
        obj, constr, x0, MMA87(); options=mma_options, convcriteria=convcriteria
    )

    simps[i] = SIMP(optimizer, solver, penalty.p);

    jooj = simps[i](x0);

end

#@show result.convstate
#@show optimizer.workspace.iter
#@show result.objval

fig = visualize(problem; topology = result[2].topology,
    default_exagg_scale = 0.07,
    scale_range = 10.0, vector_linewidth = 3, vector_arrowsize = 0.5,
)
# Makie.display(fig)