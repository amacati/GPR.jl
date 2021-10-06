using Rotations
using LinearAlgebra
using StaticArrays
using GPR
using BenchmarkTools

x = rand(10,1000)
xstar = rand(10,1)
y = rand(1,1000)

kernel = GaussianKernel(0.5, 0.5)

k = compute_kernelmatrix(x, kernel)
kstar = compute_kernelmatrix(x, xstar, kernel)
chol = cholesky!(k)
α = chol.L'\(chol.L\y')
display(kstar' * α)
display(((kstar' / chol.L') / chol.L) * y')


function fct1(kstar, chol, y)
    return (kstar'*(chol.L'\(chol.L\y')))[1]
end

function fct2(kstar, chol, y)
    return (((kstar' / chol.L') / chol.L)*y')[1]
end

@assert abs(fct1(kstar, chol, y) - fct2(kstar, chol, y)) < 1e-9

suite = BenchmarkGroup()
suite[1] = @benchmark fct1($kstar, $chol, $y)
suite[2] = @benchmark fct2($kstar, $chol, $y)
display(suite[1])
display(suite[2])
