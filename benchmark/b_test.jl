using BenchmarkTools
using LinearAlgebra
using StaticArrays
using GPR

suite = BenchmarkGroup()

function fct1(x, y)
    return x - y
end

function fct2(x, y, tmp)
    tmp = x - y
    return tmp
end

function fct3(x,y,tmp)
    tmp .= x - y
    return tmp
end

x = rand(100,100) ./ 100^2
y = rand(100,100) ./ 100^2
tmp = similar(x)

@assert sum(abs.(fct1(x,y) .- fct2(x,y, tmp))) < 1e-9
@assert sum(abs.(fct1(x,y) .- fct3(x,y, tmp))) < 1e-9

suite[1] = @benchmark fct1($x, $y)
suite[2] = @benchmark fct2($x, $y, $tmp)
suite[3] = @benchmark fct3($x, $y, $tmp)

display(suite[1])
display(suite[2])
display(suite[3])
