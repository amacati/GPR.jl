function updateF!(F::AbstractMatrix, mechanism)
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
            F[1+offset:6+offset, N+1:N+L] = Gᵥ'
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

function d(s, sᵤ, Gᵥ, Nbodies)
    return - sᵤ + s[1:Nbodies*6] + Gᵥ'*s[1+Nbodies*6:end]
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

function resetMechanism!(mechanism, states::Vector{<:ConstrainedDynamics.State})
    for id in 1:length(mechanism.bodies)
        mechanism.bodies[id].state = deepcopy(states[id])
    end
    ConstrainedDynamics.discretizestate!(mechanism)
    foreach(ConstrainedDynamics.setsolution!, mechanism.bodies)
end

function projectv!(vᵤ::Vector{<:SVector}, ωᵤ::Vector{<:SVector}, mechanism; newtonIter = 100, ϵ = 1e-10, regularizer = 0.)
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
    F += I*regularizer
    Gᵥ = (@view F[1+Nbodies*6:end, 1:Nbodies*6])
    f(s) = vcat(d(s, sᵤ, Gᵥ, Nbodies), g(mechanism))
    α = 0.5
    for _ in 1:newtonIter
        updateF!(F, mechanism)
        Δs = F\f(s)
        s -= α * Δs
        updateMechanism!(mechanism, s)
        if norm(f(s)) < ϵ && norm(g(mechanism)) < ϵ
            break
        end
    end
    return getvel(s, Nbodies)
end
