using BenchmarkTools
using LinearAlgebra
using GPR

println("Gaussian Process matrix inversion benchmark.")
suite = BenchmarkGroup()

function benchmark_gp_inversion(xtrain, ytrain, kernel)
    optimize!(GaussianProcessRegressor(xtrain, ytrain, kernel))
end

kernel = GaussianKernel(0.5,1.0)
for nsamples in [10, 100]
    local xtrain = rand(10,nsamples) .* 50  # State size 10
    local ytrain = sum(sin.(xtrain), dims=1)
    println("Matrix size: $nsamples")
    suite[nsamples] = @benchmark benchmark_gp_inversion($xtrain, $ytrain, $kernel)
    display(suite[nsamples])
end

#114μs
#18ms