using BenchmarkTools
using LinearAlgebra
using GPR
using StaticArrays
using Statistics


println("Gaussian Process prediction benchmark.")
suite = BenchmarkGroup()

kernel = GaussianKernel(0.5,1.0)
xtrain = rand(10,100) .* 5  # State size 10
ytrain = rand(10,100) .* 5

ymean = Vector{Float64}(undef, 10)
for i in 1:10
    ymean[i] = mean(ytrain[i,:])
    ytrain[i,:] .-= ymean[i]
end
gprs = [GaussianProcessRegressor(xtrain, ytrain[i,:], kernel) for i in 1:10]
ymean_static = SVector{10,Float64}(ymean)
xstart = xtrain[:,1]
xstart_static = SVector{10,Float64}(xstart)

println("Non-static prediction benchmark")
for nsteps in [100, 1000]
    println("State size: 10, number of steps: $nsteps")
    suite[nsteps] = @benchmark predict($gprs, $xstart, $nsteps, $ymean)
    display(suite[nsteps])
end

println("Static prediction benchmark")
for nsteps in [100, 1000]
    println("State size: 10, number of steps: $nsteps")
    suite[nsteps] = @benchmark predict($gprs, $xstart_static, $nsteps, $ymean_static)
    display(suite[nsteps])
end
