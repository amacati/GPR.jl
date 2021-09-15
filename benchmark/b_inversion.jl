using BenchmarkTools
using LinearAlgebra
using GPR

println("Gaussian Process matrix inversion benchmark.")
suite = BenchmarkGroup()

function benchmark_gp_inversion(xtrain, ytrain, kernel)
    GaussianProcessRegressor(xtrain, ytrain, kernel)
end

kernel = GaussianKernel(0.5,1.0)
for nsamples in [10, 100]
    local xtrain = reshape(collect(range(0, 6, length=nsamples)), 1, :)
    local ytrain = sin.(xtrain)
    println("Matrix size: $nsamples")
    suite[nsamples] = @benchmark benchmark_gp_inversion($xtrain, $ytrain, $kernel)
    display(suite[nsamples])
end
