using BenchmarkTools
using LinearAlgebra
using GPR
using StaticArrays
using Statistics


println("Gaussian Process prediction benchmark.")
suite = BenchmarkGroup()

kernel = GaussianKernel(0.5,1.0)
xtrain = rand(10,100) .* 5  # State size 10
ytrain = rand(10,100) .* 51

mo_gpr = MOGaussianProcessRegressor(xtrain, ytrain, kernel)
xstart = xtrain[:,1]
xstart_static = SVector{10,Float64}(xstart)

for nsteps in [100, 1000]
    println("State size: 10, number of steps: $nsteps")
    suite[nsteps] = @benchmark predict($mo_gpr, $xstart_static, $nsteps)
    display(suite[nsteps])
end

# 10ms
# 100ms