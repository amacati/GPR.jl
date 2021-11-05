using Rotations
using JSON
using ConstrainedDynamics: Storage, updatestate!, State, Mechanism, newton!, foreachactive, setsolution!, discretizestate!

function tostate(cstate::Vector{Float64})
    @assert length(cstate) == 13 ("State size has to be exactly 13")
    state = State{Float64}()
    state.xc = SVector(cstate[1:3]...)
    state.qc = UnitQuaternion(cstate[4], cstate[5:7])
    state.vc = SVector(cstate[8:10]...)
    state.ωc = SVector(cstate[11:13]...)
    return state
end

function tostates(cstates::Vector{Float64})
    @assert length(cstates) % 13 == 0 ("State size has to be multiple of 13")
    nstates = div(length(cstates), 13)
    states = Vector{State}(undef, nstates)
    for i in 1:nstates
        offset = (i-1)*13
        state = State{Float64}()
        state.xc = SVector(cstates[1+offset:3+offset]...)
        state.qc = UnitQuaternion(cstates[4+offset], cstates[5+offset:7+offset])
        state.vc = SVector(cstates[8+offset:10+offset]...)
        state.ωc = SVector(cstates[11+offset:13+offset]...)
        states[i] = state
    end
    return states
end

function tostates(vstates::Vector{Vector{Float64}})
    return [tostates(cstate)[1] for cstate in vstates]
end

function tocstate(state::State)
    return [state.xc..., state.qc.w, state.qc.x, state.qc.y, state.qc.z, state.vc..., state.ωc...]
end

function tocstate(vstate::Vector{Vector{Float64}})
    return reduce(vcat, vstate)
end

function tovstate(cstates::AbstractVector)
    @assert length(cstates) % 13 == 0 ("State size has to be multiple of 13")
    return [cstates[i:i+12] for i in 1:13:length(cstates)]
end

function getcstate(mechanism::Mechanism)
    Nbodies = length(mechanism.bodies)
    cstate = Vector{Float64}(undef, 13*Nbodies)
    for id in 1:Nbodies
        offset = (id-1)*13
        cstate[1+offset:13+offset] = tocstate(mechanism.bodies[id].state)
    end
    return cstate
end

function getcstate(storage::Storage, i)
    Nbodies = length(storage.x)
    cstate = Vector{Float64}(undef, 13*Nbodies)
    for id in 1:Nbodies
        offset = (id-1)*13
        cstate[1+offset:13+offset] = [storage.x[id][i]..., storage.q[id][i].w, storage.q[id][i].x, storage.q[id][i].y, storage.q[id][i].z,
        storage.v[id][i]..., storage.ω[id][i]...]
    end
    return cstate
end

function getvstates(storage::Storage, i)
    Nbodies = length(storage.x)
    return [[storage.x[id][i]..., storage.q[id][i].w, storage.q[id][i].x, storage.q[id][i].y, storage.q[id][i].z,
             storage.v[id][i]..., storage.ω[id][i]...] for id in 1:Nbodies]
end

function getvstates(mechanism::Mechanism)
    Nbodies = length(mechanism.bodies)
    return [tocstate(mechanism.bodies[id].state) for id in 1:Nbodies]
end

function getstates(mechanism::Mechanism)
    Nbodies = length(mechanism.bodies)
    return [deepcopy(mechanism.bodies[id].state) for id in 1:Nbodies]
end

function setstates!(mechanism::Mechanism, vstates::Vector{Vector{Float64}})
    @assert length(vstates) == length(mechanism.bodies) ("State vector size has to be #Nbodies!") 
    states = tostates(vstates)
    for id in 1:length(mechanism.bodies)
        mechanism.bodies[id].state = states[id]
    end
    discretizestate!(mechanism)
    foreach(setsolution!, mechanism.bodies)
end

function overwritestorage(storage::Storage, states, i)
    @assert length(states) == length(storage.x) ("State vector size has to be #Nbodies!") 
    for id in 1:length(states)
        storage.x[id][i] = states[id][1:3]
        storage.q[id][i] = UnitQuaternion(states[id][4], states[id][5:7])
        storage.v[id][i] = states[id][8:10]
        storage.ω[id][i] = states[id][11:13]
    end
end

function simulationerror(groundtruth::Storage, predictions::Storage; stop::Integer = length(predictions.x[1]))
    @assert 1 < stop <= length(predictions.x[1])
    @assert length(groundtruth.x[1]) >= length(predictions.x[1])
    Nbodies = length(groundtruth.x)
    error = 0
    for i in 2:stop
        # get vector, compute error
        xtrue = [state[1:3] for state in getvstates(groundtruth, i)]  # for t+1
        xpred = [state[1:3] for state in getvstates(predictions, i)]  # for t+1
        for id in 1:Nbodies
            error += sum((xtrue[id] .- xpred[id]).^2)
        end
    end
    error /= (3*Nbodies*(stop-1))
    isnan(error) ? (return Inf) : (return error)
end

function simulationerror(groundtruth::Vector{<:Vector}, predictions::Vector{<:Vector}; stop::Integer = length(predictions))
    @assert 1 < stop <= length(predictions)
    @assert length(groundtruth[1]) % 13 == 0 ("State has to be of length Nbodies*13")
    @assert length(groundtruth) >= length(predictions)
    Nbodies = div(length(groundtruth[1]), 13)
    error = 0
    for i in 1:stop
        # get vector, compute error
        xtrue = [state[1:3] for state in tovstate(groundtruth[i])]  # for t+1
        xpred = [state[1:3] for state in tovstate(predictions[i])]  # for t+1
        for id in 1:Nbodies
            error += sum((xtrue[id] .- xpred[id]).^2)
        end
    end
    error /= (3*Nbodies*(stop))
    isnan(error) ? (return Inf) : (return error)
end

function savecheckpoint(experimentid::String, checkpointdict::Dict)
    open(joinpath(dirname(@__FILE__), "data", experimentid*"_checkpoint.json"),"w") do f
        JSON.print(f, checkpointdict)
    end
end

function loadcheckpoint(experimentid::String)
    open(joinpath(dirname(@__FILE__), "data", experimentid*"_checkpoint.json"),"r") do f
        return JSON.parse(f)
    end
end
