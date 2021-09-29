using BenchmarkTools
using LinearAlgebra
using StaticArrays
using GPR

suite = BenchmarkGroup()

function fct1(x::Float64)
    if x % 2 == 0
        return x
    end
    return x, x
end

function fct2(x::Float64)
    return x, x
end

const x1 = 143.2
suite[1] = @benchmark fct1($x1)
suite[2] = @benchmark fct2($x1)

display(suite[1])
display(suite[2])
