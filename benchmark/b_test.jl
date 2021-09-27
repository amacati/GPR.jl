using BenchmarkTools
using LinearAlgebra
using StaticArrays


suite = BenchmarkGroup()

function fct1(x::Float64)
    return x^3
end

function fct2(x1::Float64, x2::Float64)
    return x1 * x2
end

const x1 = 143.2
const x2 = x1^2
suite[1] = @benchmark fct1($x1)
suite[2] = @benchmark fct2($x1, $x2)

display(suite[1])
display(suite[2])
