using ConstrainedDynamics: State, Storage, Mechanism

struct CState{T,N} <: AbstractVector{T}
    
    state::SVector

    function CState(x::AbstractVector)
        @assert length(x) % 13 == 0 && length(x) > 0 ("CState needs Nbodies*13 entries!")
        T = typeof(x[1])
        N = div(length(x), 13)
        new{T,N}(SVector{length(x), T}(x))
    end

    function CState{T,N}(x::AbstractVector) where {T,N}
        @assert length(x) == 13*N ("CState needs Nbodies*13 entries!")
        new{T,N}(SVector{length(x), T}(x))
    end

    function CState(s::State)
        new{Float64,1}(SVector{13, Float64}([s.xc..., s.qc.w, s.qc.x, s.qc.y, s.qc.z, s.vc..., s.ωc...]))
    end

    function CState(sv::Vector{<:State})
        N = length(sv)
        T = Float64
        x = reduce(vcat, [[s.xc..., s.qc.w, s.qc.x, s.qc.y, s.qc.z, s.vc..., s.ωc...] for s in sv])
        new{T,N}(SVector{13*N, T}(x))
    end
end

Base.size(cstate::CState) = return size(cstate.state)

Base.getindex(cstate::CState, i::Int) = return getindex(cstate.state, i)

Base.setindex!(cstate::CState{T,N}, v::T, i::Int) where {T,N} = setindex!(cstate.state, v, i)

function toStates(cstate::CState{T,N}) where {T,N}
    states = Vector{State}(undef, N)
    for i in 1:N
        offset = (i-1)*13
        state = State{Float64}()
        state.xc = SVector{3,T}(cstate[1+offset:3+offset])
        state.qc = UnitQuaternion(cstate[4+offset], cstate[5+offset:7+offset])
        state.vc = SVector{3,T}(cstate[8+offset:10+offset])
        state.ωc = SVector{3,T}(cstate[11+offset:13+offset])
        states[i] = state
    end
    return states
end

function toState(cstate::CState{T,1}) where {T}
    state = State{Float64}()
    state.xc = SVector{3,T}(cstate[1:3])
    state.qc = UnitQuaternion(cstate[4], cstate[5:7])
    state.vc = SVector{3,T}(cstate[8:10])
    state.ωc = SVector{3,T}(cstate[11:13])
    return state
end

function CState(mechanism::Mechanism; usesolution=false)
    usesolution && return _CStateSolution(mechanism)
    N = length(mechanism.bodies)
    return CState([mechanism.bodies[id].state for id in 1:N])
end

function _CStateSolution(mechanism::Mechanism)
    N = length(mechanism.bodies)
    sv = [mechanism.bodies[i].state for i in 1:N]
    x = reduce(vcat, [[s.xsol[2]..., q2vec(s.qsol[2])..., s.vsol[2]..., s.ωsol[2]...] for s in sv])
    return CState(SVector{13*N, Float64}(x))
end

function CState(storage::Storage{T,N}, i::Int) where {T,N}
    Nb = length(storage.x)
    x = Vector{T}(undef, Nb*13)
    for id in 1:Nb
        offset = (id-1)*13
        x[1+offset:3+offset] = storage.x[id][i]
        x[4+offset:7+offset] = q2vec(storage.q[id][i])
        x[8+offset:10+offset] = storage.v[id][i]
        x[11+offset:13+offset] = storage.ω[id][i]
    end
    return CState(x)
end