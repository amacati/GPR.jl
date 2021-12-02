q2vec(q::UnitQuaternion) = return [q.w, q.x, q.y, q.z]

getStates(mechanism::Mechanism) = return [deepcopy(mechanism.bodies[id].state) for id in 1:length(mechanism.bodies)]
