using Rotations
using JSON
using ConstrainedDynamics: Storage, updatestate!, State, Mechanism, newton!, foreachactive, setsolution!, discretizestate!

function vector2state(vstate)
    @assert length(vstate) == 13 ("State size has to be 13")
    state = State{Float64}()
    state.xc = SVector(vstate[1:3]...)
    state.qc = UnitQuaternion(vstate[4], vstate[5:7])
    state.vc = SVector(vstate[8:10]...)
    state.ωc = SVector(vstate[11:13]...)
    return state
end

function state2vector(state::State)
    return [state.xc..., state.qc.w, state.qc.x, state.qc.y, state.qc.z, state.vc..., state.ωc...]
end

function cstate2state(cstate::Vector{Float64})
    @assert length(cstate) % 13 == 0
    return [cstate[1+offset:13+offset] for offset in 0:13:length(cstate)-1]
end

function getstates(storage::Storage, i)
    Nbodies = length(storage.x)
    return [[storage.x[id][i]..., storage.q[id][i].w, storage.q[id][i].x, storage.q[id][i].y, storage.q[id][i].z,
             storage.v[id][i]..., storage.ω[id][i]...] for id in 1:Nbodies]
end

function getstates(mechanism::Mechanism)
    Nbodies = length(mechanism.bodies)
    return [state2vector(mechanism.bodies[id].state) for id in 1:Nbodies]
end

function setstates!(mechanism, states)
    @assert length(states) == length(mechanism.bodies) ("State vector size has to be #Nbodies!") 
    for id in 1:length(mechanism.bodies)
        mechanism.bodies[id].state = vector2state(states[id])
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
        xtrue = [state[1:3] for state in getstates(groundtruth, i)]  # for t+1
        xpred = [state[1:3] for state in getstates(predictions, i)]  # for t+1
        for id in 1:Nbodies
            error += sum((xtrue[id] .- xpred[id]).^2)
        end
    end
    error /= (3*Nbodies*(stop-1))
    isnan(error) ? (return Inf) : (return error)
end

function simulationerror(groundtruth::Vector{<:Vector}, predictions::Vector{<:Vector}; stop::Integer = length(predictions))
    @assert 1 < stop <= length(predictions)
    @assert length(groundtruth) >= length(predictions)
    Nbodies = length(mechanism.bodies)
    error = 0
    for i in 1:stop
        # get vector, compute error
        xtrue = [state[1:3] for state in cstate2state(groundtruth[i])]  # for t+1
        xpred = [state[1:3] for state in cstate2state(predictions[i])]  # for t+1
        for id in 1:Nbodies
            error += sum((xtrue[id] .- xpred[id]).^2)
        end
    end
    error /= (3*Nbodies*(stop-1))
    isnan(error) ? (return Inf) : (return error)
end

function checkpoint(experimentid::String, checkpointdict::Dict)
    open(experimentid*"_checkpoint.json","w") do f
        JSON.print(f, checkpointdict)
    end
end

function loadcheckpoint(experimentid::String)
    checkpointdict = Dict()
    # If there is no checkpoint, default to empty dictionary
    !Base.Filesystem.isfile(experimentid*"_checkpoint.json") && (return false, checkpointdict)
    open(experimentid*"_checkpoint.json","r") do f
        checkpointdict = JSON.parse(f)
    end
    return true, checkpointdict
end
