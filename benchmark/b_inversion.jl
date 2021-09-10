using BenchmarkTools
using LinearAlgebra
include("../src/GaussianProcessRegressor.jl")

println("Gaussian Process matrix inversion benchmark.")
suite = BenchmarkGroup()

function benchmark_gp_inversion(xtrain, ytrain, kernel)
    GPR.GaussianProcessRegressor(xtrain, ytrain, kernel)
end

kernel = GPR.GaussianKernel(0.5,1.0)
for nsamples in [10, 100]
    local xtrain = reshape(collect(range(0, 6, length=nsamples)), 1, :)
    local ytrain = sin.(xtrain)
    println("Matrix size: $nsamples")
    suite[nsamples] = @benchmark benchmark_gp_inversion($xtrain, $ytrain, $kernel)
    display(suite[nsamples])
end
