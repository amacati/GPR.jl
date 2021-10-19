using BenchmarkTools
using LinearAlgebra
using StaticArrays
using GPR

suite = BenchmarkGroup()

function fct1(input)
    return SVector(input...)
end

function fct2(input)
    return SVector{length(input), Float64}(input)
end

input = rand(20)

suite[1] = @benchmark fct1($input)
suite[2] = @benchmark fct2($input)

display(suite[1])
display(suite[2])
