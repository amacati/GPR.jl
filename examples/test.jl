using Rotations
using LinearAlgebra
using StaticArrays
using GPR
using BenchmarkTools
using ConstrainedDynamics

include("generatedata.jl")


function updateF!(F::AbstractMatrix, mechanism; Gᵥonly = false)
    Gᵥonly ? (_updateFv!(F, mechanism)) : (_updateFxv!(F, mechanism))
end

# Only update Gᵥ in [1 Gₓ; Gᵥ 0]
function _updateFv!(F::AbstractMatrix, mechanism)
    N = length(mechanism.bodies)*6
    for eqc in mechanism.eqconstraints
        L = length(eqc)
        ids = unique(eqc.childids)
        if eqc.parentid !== nothing
            push!(ids, eqc.parentid) 
        end
        for id in ids
            Gᵥ = ConstrainedDynamics.∂g∂ʳvel(mechanism, eqc, id)
            F[N+1:N+L, 1+(id-1)*6:id*6] = Gᵥ
        end
        N += L  # Advance along first dimension
    end
    return nothing
end

# Update both Gₓ and Gᵥ in [1 Gₓ; Gᵥ 0]
function _updateFxv!(F::AbstractMatrix, mechanism)
    N = length(mechanism.bodies)*6
    for eqc in mechanism.eqconstraints
        L = length(eqc)
        ids = unique(eqc.childids)
        if eqc.parentid !== nothing
            push!(ids, eqc.parentid) 
        end
        for id in ids
            Gₓ = ConstrainedDynamics.∂g∂ʳpos(mechanism, eqc, id)
            Gᵥ = ConstrainedDynamics.∂g∂ʳvel(mechanism, eqc, id)
            F[N+1:N+L, 1+(id-1)*6:id*6] = Gᵥ
            F[1+(id-1)*6:id*6, N+1:N+L] = Gₓ'
        end
        N += L  # Advance along first dimension
    end
    return nothing
end

function updateS!(s, v, ω)
    N = length(v)
    for (iv, is) in zip(1:N, 1:6:N*6)
        s[is:is+2] = v[iv]
        s[is+3:is+5] = ω[iv]
    end
end

function updateMechanism!(mechanism::Mechanism, v::Vector{<:SVector}, ω::Vector{<:SVector})
    N = length(v)  # Number of positions/bodies to updatemechanism
    @assert N == length(mechanism.bodies) ("Number of update velocities must equal the number of bodies of the mechanism!")
    for i in 1:N
        mechanism.bodies[i].state.vsol[2] = v[i]
        mechanism.bodies[i].state.ωsol[2] = ω[i]
        ConstrainedDynamics.updatestate!(mechanism.bodies[i], mechanism.Δt)
    end
end

function getvel(s, Nbodies)
    v, ω = Vector{SVector{3, Float64}}(), Vector{SVector{3, Float64}}()
    for sid in 1:6:Nbodies*6
        push!(v, SVector{3, Float64}(s[sid:sid+2]))
        push!(ω, SVector{3, Float64}(s[sid+3:sid+5]))
    end
    return v, ω
end

function d(s, sᵤ, Gₓ, Nbodies)
    return sᵤ - s[1:Nbodies*6] + Gₓ'*s[1+Nbodies*6:end]
end

function g(mechanism, Ndims)
    g = zeros(MVector{Ndims})
    N = 1
    for eqc in mechanism.eqconstraints
        L = length(eqc)
        g[N:N+L-1] = ConstrainedDynamics.g(mechanism, eqc)
        N += L
    end
    return g
end

function projectv!(vᵤ, ωᵤ, newtonIter, mechanism)
    Ndims = sum([length(eqc) for eqc in mechanism.eqconstraints])  # Total dimensionality of constraints
    Nbodies = length(mechanism.bodies)
    F = zeros(Nbodies*6 + Ndims, Nbodies*6 + Ndims)  # 3 vel, 3 ω for each body -> 6
    for i in 1:Nbodies*6 F[i,i] = 1 end
    updateF!(F, mechanism)
    Gₓ = (F[1:Nbodies*6, 1+Nbodies*6:end])'
    s = zeros(MVector{6*Nbodies + Ndims})
    sᵤ = zeros(MVector{6*Nbodies})
    updateS!(s, vᵤ, ωᵤ)  # Initial s is [v1, ω1, v2, ω2, ...,  λ1, λ2, ...] with λ = 0
    updateS!(sᵤ, vᵤ, ωᵤ)
    v, ω = vᵤ, ωᵤ  # Initialize first guess with unconstrained values
    display(mechanism.bodies[1].state.xc)
    updateMechanism!(mechanism, v, ω)
    f(s) = vcat(d(s, sᵤ, Gₓ, Nbodies), g(mechanism, Ndims))
    for _ in 1:newtonIter
        updateF!(F, mechanism, Gᵥonly=true)
        Δs = F\f(s)
        s -= Δs
        v, ω = getvel(s, Nbodies)
        updateMechanism!(mechanism, v, ω)
    end
    display(mechanism.bodies[1].state.xc)
    return v, ω
end

storage, mechanism = simplependulum2D()
v = [storage.v[1][2]]
ω = [storage.ω[1][2]]
display(projectv!(v, ω, 10, mechanism))

display(storage.x)