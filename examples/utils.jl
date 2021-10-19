using ConstrainedDynamics: Storage

function getstate(mechanism)
    Nbodies = length(mechanism.bodies)
    states = [[mechanism.bodies[id].state.xc..., mechanism.bodies[id].state.qc.w, mechanism.bodies[id].state.qc.x, mechanism.bodies[id].state.qc.y,
               mechanism.bodies[id].state.qc.z, mechanism.bodies[id].state.vc..., mechanism.bodies[id].state.ωc...]
              for id in 1:Nbodies]
    state = SVector(reduce(vcat, states)...)
    return state
end

function overwritestorage(storage::Storage, state, idx)
    Nbodies = length(storage.x)
    for id in 1:Nbodies
        offset = (id-1)*13
        storage.x[id][idx] = state[1+offset:3+offset]
        storage.q[id][idx] = UnitQuaternion(state[4+offset], state[5+offset:7+offset])
    end
end
