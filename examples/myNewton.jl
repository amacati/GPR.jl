using LinearAlgebra
using ConstrainedDynamics


function myDynamics!(mechanism, body, offset, s)
    state = body.state
    Δt = mechanism.Δt
    gravity = SVector(0., 0, -mechanism.g)
    dynT = body.m * ((state.vsol[2] - state.vc) / Δt + gravity) - state.Fk[1]
    ω1 = state.ωc
    ω2 = state.ωsol[2]
    J = body.J
    sq1 = sqrt(4 / Δt^2 - ω1' * ω1)
    sq2 = sqrt(4 / Δt^2 - ω2' * ω2)
    dynR = ConstrainedDynamics.skewplusdiag(ω2, sq2) * (J * ω2) - ConstrainedDynamics.skewplusdiag(ω1, sq1) * (J * ω1) - 2*state.τk[1]
    s[1+offset:3+offset] = dynT
    s[4+offset:6+offset] = dynR
end

function updateF!(F::AbstractMatrix, mechanism; Gᵥonly = false)
    Gᵥonly ? (_updateFv!(F, mechanism)) : (_updateF!(F, mechanism))
end

# Only update Gᵥ in [D Gₓ; Gᵥ 0]
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

# Update everything in [D Gₓ; Gᵥ 0]
function _updateF!(F::AbstractMatrix, mechanism)
    for (id, body) in enumerate(mechanism.bodies)
        offset = (id-1)*6
        F[1+offset:6+offset, 1+offset:6+offset] = ConstrainedDynamics.∂dyn∂vel(mechanism, body)
    end
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

function myUpdateMechanism!(mechanism, s)
    for (id, body) in enumerate(mechanism.bodies)
        state = body.state
        offset = (id-1)*6
        state.vsol[2] = s[1+offset:3+offset]
        state.ωsol[2] = s[4+offset:6+offset]
    end
    Nbodies = length(mechanism.bodies)
    offset = Nbodies
    for eqc in mechanism.eqconstraints
        eqcdim = length(eqc)
        eqc.λsol[2] = s[1+offset:eqcdim+offset]
        offset += eqcdim
    end
end

function d(mechanism, s)
    Nbodies = length(mechanism.bodies)
    myUpdateMechanism!(mechanism, s)
    d = zeros(MVector{Nbodies*6})
    for (id, body) in enumerate(mechanism.bodies)
        myDynamics!(mechanism, body, (id-1)*6, d)
    end
    return d
end

function g(mechanism)
    Ndims = sum([length(eqc) for eqc in mechanism.eqconstraints])  # Total dimensionality of constraints
    g = zeros(MVector{Ndims})
    N = 1
    for eqc in mechanism.eqconstraints
        L = length(eqc)
        g[N:N+L-1] = ConstrainedDynamics.g(mechanism, eqc)
        N += L
    end
    return g
end

function _myNewton!(mechanism, s; ϵ = 1e-10, newtonIter = 100)
    Ndims = sum([length(eqc) for eqc in mechanism.eqconstraints])  # Total dimensionality of constraints
    Nbodies = length(mechanism.bodies)
    F = zeros(Nbodies*6 + Ndims, Nbodies*6 + Ndims)  # 3 vel, 3 ω for each body -> 6
    myUpdateMechanism!(mechanism, s)
    updateF!(F, mechanism)
    f(s) = vcat(d(mechanism, s), g(mechanism))
    oldΔs = zeros(MVector{6*Nbodies + Ndims})
    for i in 1:newtonIter
        updateF!(F, mechanism, Gᵥonly=true)
        Δs = F\f(s)
        s -= 0.5 * Δs
        myUpdateMechanism!(mechanism, s)
        normΔs = sum((Δs .- oldΔs).^2)
        oldΔs = Δs
        if normΔs < ϵ
            break
        end
    end
    foreach(ConstrainedDynamics.updatestate!, mechanism.bodies, mechanism.Δt)
end