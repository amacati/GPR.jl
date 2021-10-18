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
        storage.x[id][idx] = state[1+((id-1)*13):3+((id-1)*13)]
        storage.q[id][idx] = UnitQuaternion(state[4+((id-1)*13)], state[5+((id-1)*13):7+((id-1)*13)])
    end
end
