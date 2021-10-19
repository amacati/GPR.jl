using LinearAlgebra
using ConstrainedDynamics

function updateF!(F::AbstractMatrix, mechanism; Gᵥonly = false)
    Gᵥonly ? (_updateFv!(F, mechanism)) : (_updateF!(F, mechanism))
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
            offset = (id-1)*6
            F[N+1:N+L, 1+offset:6+offset] = Gᵥ
        end
        N += L  # Advance along first dimension
    end
    return nothing
end

# Update both Gₓ and Gᵥ in [1 Gₓ; Gᵥ 0]
function _updateF!(F::AbstractMatrix, mechanism)
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
            offset = (id-1)*6
            F[N+1:N+L, 1+offset:6+offset] = Gᵥ
            F[1+offset:6+offset, N+1:N+L] = Gₓ'
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

function updateMechanism!(mechanism::Mechanism, s)
    for (id, body) in enumerate(mechanism.bodies)
        state = body.state
        offset = (id-1)*6
        state.vsol[2] = s[1+offset:3+offset]
        state.ωsol[2] = s[4+offset:6+offset]
    end
    Nbodies = length(mechanism.bodies)
    offset = Nbodies
    for eqc in mechanism.eqconstraints
        eqcdim = length(eqc.λsol[1])
        eqc.λsol[2] = s[1+offset:eqcdim+offset]
        offset += eqcdim
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

function g(mechanism)
    Ndims = sum([length(eqc) for eqc in mechanism.eqconstraints])
    g = zeros(MVector{Ndims})
    N = 1
    for eqc in mechanism.eqconstraints
        L = length(eqc)
        g[N:N+L-1] = ConstrainedDynamics.g(mechanism, eqc)
        N += L
    end
    return g
end

function resetMechanism!(mechanism, states; overwritesolution = false)
    for id in 1:length(mechanism.bodies)
        mechanism.bodies[id].state = deepcopy(states[id])
        if overwritesolution
            mechanism.bodies[id].state.vsol[2] = states[id].vc
            mechanism.bodies[id].state.ωsol[2] = states[id].ωc
            mechanism.bodies[id].state.xk[1] = states[id].xc + states[id].vc*mechanism.Δt
            mechanism.bodies[id].state.qk[1] = states[id].qc * ConstrainedDynamics.ωbar(states[id].ωc,mechanism.Δt) * mechanism.Δt / 2
            mechanism.bodies[id].state.xsol[2] = mechanism.bodies[id].state.xk[1]
            mechanism.bodies[id].state.qsol[2] = mechanism.bodies[id].state.qk[1]
        end
    end
    # foreach(setsolution!, mechanism.bodies)
end

function projectv!(vᵤ::Vector{<:SVector}, ωᵤ::Vector{<:SVector}, mechanism; newtonIter = 100, ϵ = 1e-10)
    Ndims = sum([length(eqc) for eqc in mechanism.eqconstraints])  # Total dimensionality of constraints
    Nbodies = length(mechanism.bodies)
    F = zeros(Nbodies*6 + Ndims, Nbodies*6 + Ndims)  # 3 vel, 3 ω for each body -> 6
    for i in 1:Nbodies*6 F[i,i] = 1 end
    s = zeros(MVector{6*Nbodies + Ndims})
    sᵤ = zeros(MVector{6*Nbodies})
    updateS!(s, vᵤ, ωᵤ)  # Initial s is [v1, ω1, v2, ω2, ...,  λ1, λ2, ...] with λ = 0
    updateS!(sᵤ, vᵤ, ωᵤ)
    updateMechanism!(mechanism, s)
    updateF!(F, mechanism)
    Gₓ = (F[1:Nbodies*6, 1+Nbodies*6:end])'
    v, ω = vᵤ, ωᵤ  # Initialize first guess with unconstrained values
    f(s) = vcat(d(s, sᵤ, Gₓ, Nbodies), g(mechanism))
    oldΔs = zeros(MVector{6*Nbodies + Ndims})
    for i in 1:newtonIter
        updateF!(F, mechanism, Gᵥonly=true)
        Δs = F\f(s)
        s -= 0.5/(i) * Δs
        updateMechanism!(mechanism, s)
        normΔs = sum((Δs - oldΔs).^2)
        if normΔs < ϵ
            break
        end
    end
    foreach(ConstrainedDynamics.updatestate!, mechanism.bodies, mechanism.Δt)
    return getvel(s, Nbodies)
end