using Rotations
using LinearAlgebra
using StaticArrays
using GPR
using BenchmarkTools
using ConstrainedDynamics

include("generatedata.jl")


function constructF(mechanism, eqc, body)
    Gₓ = ConstrainedDynamics.∂g∂ʳpos(mechanism, eqc, body.id)
    Gᵥ = ConstrainedDynamics.∂g∂ʳvel(mechanism, eqc, body.id)
    F = vcat(hcat(I, Gₓ'), hcat(Gᵥ, zeros(size(Gᵥ, 1), size(Gᵥ, 1))))  # [I Gₓ'; Gᵥ 0]
    return F
end

function projectv!(v, newtonIter, mechanism, eqc, body)
    F = constructF(mechanism, eqc, body)
    f(s, G) = v - s[1:3] + G's[4:end]
    s = SVector(v..., zeros(size(F, 1)-3)...)  # Initial s is [v, λ] with v = v, λ = 0
    for _ in 1:newtonIter
        F = constructF(mechanism, eqc, body)
        Δs = F\f(s, F)
        s -= Δs
    end
    return s[1:3]
end

storage, mechanism = simplependulum3D()
F = constructF(mechanism, mechanism.eqconstraints[2], mechanism.bodies[1])
display(F)
