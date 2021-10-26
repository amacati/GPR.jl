using Rotations
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

function getstates(storage::Storage, i)
    Nbodies = length(storage.x)
    return [[storage.x[id][i]..., storage.q[id][i].w, storage.q[id][i].x, storage.q[id][i].y, storage.q[id][i].z,
             storage.v[id][i]..., storage.ω[id][i]...] for id in 1:Nbodies]
end

function getstates(mechanism::Mechanism)
    Nbodies = length(mechanism.bodies)
    return [state2vector(mechanism.bodies[id].state) for id in 1:Nbodies]
end

function setstates(mechanism, states)
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

function onesteperror(mechanism, predictions::Storage; stop = length(predictions.x[1])-1)
    @assert stop < length(predictions.x[1])
    tmpstorage = Storage{Float64}(1000, length(mechanism.bodies))
    Nbodies = length(mechanism.bodies)
    error = 0
    for i in 2:stop
        # set state of each body. also reverts previous changes in mechanism
        initialstates = getstates(predictions, i-1)
        setstates(mechanism, initialstates)
        # make one step update
        newton!(mechanism)
        foreachactive(updatestate!, mechanism.bodies, mechanism.Δt)
        # get vector, compute error
        xtrue = [mechanism.bodies[id].state.xsol[2] for id in 1:Nbodies]  # for t+1
        xpred = [state[1:3] for state in getstates(predictions, i+1)]  # for t+1
        for id in 1:Nbodies
            error += sum((xtrue[id] .- xpred[id]).^2)
        end
    end
    error /= (3*Nbodies*stop)
    return error
end
