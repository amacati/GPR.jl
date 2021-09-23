using BenchmarkTools
using LinearAlgebra
using GPR
using StaticArrays

println("Gaussian Process inference benchmark.")
suite = BenchmarkGroup()

kernel = GaussianKernel(0.5,1.0)
xtrain = rand(10,100) .* 5  # State size 10
ytrain = sum(sin.(xtrain), dims=1)
gpr = GaussianProcessRegressor(xtrain, ytrain, kernel, 0.05)

for nsamples in [10, 100]
    local xtest = [SVector{10}(rand(10)) for _ in 1:nsamples]  
    println("State size: 10, inference size: $nsamples")
    suite[nsamples] = @benchmark predict($gpr, $xtest)
    display(suite[nsamples])
end

#166 μs
#1.9 ms