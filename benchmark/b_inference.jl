using BenchmarkTools
using LinearAlgebra
using GPR
using StaticArrays

println("Gaussian Process inference benchmark.")
suite = BenchmarkGroup()

kernel = GaussianKernel(0.5,1.0)
xtrain = [SVector{10, Float64}(rand(10) .* 5) for _ in 1:100]  # State size 10
ytrain = [sum(sin.(sample)) for sample in xtrain]
gpr = GaussianProcessRegressor(xtrain, ytrain, kernel)

for nsamples in [10, 100]
    local xtest = [SVector{10}(rand(10)) for _ in 1:nsamples]  
    println("State size: 10, inference size: $nsamples")
    suite[nsamples] = @benchmark predict($gpr, $xtest)
    display(suite[nsamples])
end
