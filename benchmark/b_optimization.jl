using BenchmarkTools
using StaticArrays
using LinearAlgebra
using GPR

println("Gaussian Process optimization benchmark.")
suite = BenchmarkGroup()

function benchmark_gp_inversion(xtrain, ytrain, kernel)
    optimize!(GaussianProcessRegressor(xtrain, ytrain, kernel))
end

for nsamples in [10, 100]
    xtrain = [SVector{10, Float64}(rand(10) .* 5) for _ in 1:nsamples]  # State size 10
    ytrain = [sum(sin.(sample)) for sample in xtrain]
    # kernel = GaussianKernel(0.5,1.0)
    # kernel = GeneralGaussianKernel(0.5,ones(10))
    kernel = CompositeKernel([GaussianKernel(0.5,1.0), GaussianKernel(0.5,1.0)], [5, 5])
    println("Sample size: $nsamples")
    suite[nsamples] = @benchmark benchmark_gp_inversion($xtrain, $ytrain, $kernel)
    display(suite[nsamples])
end
