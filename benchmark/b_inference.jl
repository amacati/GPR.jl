using BenchmarkTools
using LinearAlgebra
using GPR

println("Gaussian Process inference benchmark.")
suite = BenchmarkGroup()

kernel = GaussianKernel(0.5,1.0)
xtrain = rand(10,100) .* 5  # State size 10
ytrain = sin.(xtrain)
gpr = GaussianProcessRegressor(xtrain, ytrain, kernel, 0.05)

for nsamples in [10, 100]
    local xtest = rand(10, nsamples) .* 5
    println("State size: 10, inference size: $nsamples")
    suite[nsamples] = @benchmark GPR.predict($gpr, $xtest)
    display(suite[nsamples])
end
