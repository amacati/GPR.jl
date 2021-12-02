using Rotations
using JSON
using ConstrainedDynamics: Mechanism, Storage, State, updatestate!, newton!, foreachactive, setsolution!, discretizestate!


q2vec(q::UnitQuaternion) = return [q.w, q.x, q.y, q.z]

getStates(mechanism::Mechanism) = return [deepcopy(mechanism.bodies[id].state) for id in 1:length(mechanism.bodies)]

function getStates(storage::Storage, i::Int)
    N = length(storage.x)
    states = Vector{State{Float64}}(undef, N)
    for id in 1:N
        state = State{Float64}()
        state.xc = storage.x[id][i]
        state.qc = storage.q[id][i]
        state.vc = storage.v[id][i]
        state.ωc = storage.ω[id][i]
        states[id] = state
    end
    return states
end

function setstates!(mechanism::Mechanism, cstate::CState{T,N}) where {T,N}
    @assert N == length(mechanism.bodies) ("CState bodies don't match mechanism!") 
    states = toStates(cstate)
    for id in 1:N
        mechanism.bodies[id].state = states[id]
    end
    discretizestate!(mechanism)
    foreach(setsolution!, mechanism.bodies)
end

function overwritestorage(storage::Storage, cstate::CState{T,N}, i::Int) where {T,N}
    @assert N == length(storage.x) ("CState bodies don't match the storage!") 
    for id in 1:N
        offset = (id-1)*13
        storage.x[id][i] = cstate[1+offset:3+offset]
        storage.q[id][i] = UnitQuaternion(cstate[4+offset], cstate[5+offset:7+offset])
        storage.v[id][i] = cstate[8+offset:10+offset]
        storage.ω[id][i] = cstate[11+offset:13+offset]
    end
end

function simulationerror(groundtruth::Vector{CState{T,N}}, predictions::Vector{CState{T,N}}; stop::Int = length(predictions)) where {T,N}
    @assert 1 <= stop <= length(predictions)
    @assert length(groundtruth) == length(predictions) ("Prediction sample size does not match ground truth!")
    error = 0
    for i in 1:stop, id in 1:N
        offset = (id-1)*13
        error += sum((groundtruth[i][1+offset:3+offset] - predictions[i][1+offset:3+offset]).^2)
    end
    error /= (3N*(stop))
    isnan(error) ? (return Inf) : (return error)
end

function savecheckpoint(experimentid::String, checkpointdict::Dict)
    path = joinpath(dirname(dirname(@__FILE__)), "results", experimentid*"_checkpoint.json")
    if !Base.Filesystem.isfile(path)
        Base.Filesystem.mkpath(dirname(path))
        Base.Filesystem.touch(path)
    end
    open(path, "w") do f
        JSON.print(f, checkpointdict)
    end
end

function loadcheckpoint(experimentid::String)
    open(joinpath(dirname(dirname(@__FILE__)), "results", experimentid*"_checkpoint.json"),"r") do f
        return JSON.parse(f)
    end
end
