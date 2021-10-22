using LinearAlgebra
using Random
using GPR
using Plots

pyplot()


function plot_gp(xlim::Tuple{Real, Real}, ylim::Tuple{Real, Real}, f::Function, ftrue::Function)
    pyplot()
    plot(plot3d(range(xlim..., length=100), range(ylim..., length=100), f, st=:surface, camera=(-30,30),
                xlabel = "Feature 1", ylabel = "Feature 2", zlabel = "GP interpolation"),
         plot3d(range(xlim..., length=100), range(ylim..., length=100), ftrue, st=:surface, camera=(-30,30),
                xlabel = "Feature 1", ylabel = "Feature 2", zlabel = "Ground truth"))
end



x1 = collect(repeat(range(-5, length=10, stop=5), outer=10))
x2 = collect(repeat(range(-5, length=10, stop=5), inner=10))

xtrain = [SVector(x1i,x2i) for (x1i, x2i) in zip(x1, x2)]
ytrain = [sin.(xi[1]) .* cos.(xi[2]) for xi in xtrain]

kernel = GaussianKernel(2, 0.2)
kernel = GeneralGaussianKernel(2, ones(2)*4)
kernel = CompositeKernel([GaussianKernel(2, 0.2), GaussianKernel(2, 0.2)], [1,1])
regressor = GaussianProcessRegressor(xtrain, ytrain, kernel)
println("Initial Marginal Likelihood: $(regressor.log_marginal_likelihood)")  # -164.29517869120883
optimize!(regressor)

if typeof(kernel) == GaussianKernel
    println("Log Marginal Likelihood: $(regressor.log_marginal_likelihood), σ:$(regressor.kernel.σ) λ:$(regressor.kernel.λ)")
elseif typeof(kernel) == GeneralGaussianKernel
    println("Log Marginal Likelihood: $(regressor.log_marginal_likelihood), σ:$(regressor.kernel.σ) Λ:$(regressor.kernel.Λ)")
end
# Visualization needs callables because of the python interface
f(x,y) = predict(regressor, SVector(x,y))[1][1]  # First tuple + vector element
ftrue(x,y) = sin(x) * cos(y)
plot_gp((-5,5), (-5,5), f, ftrue)

# 0.008
