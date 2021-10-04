using Rotations
using LinearAlgebra


axis = [1, 0, 0]
θ = pi/2

display("###"^6)
x1 = UnitQuaternion(AngleAxis(θ, axis...))
x2 = UnitQuaternion(AngleAxis(θ/2, axis...))

function constructQ(quats::Vector{UnitQuaternion{Float64}})
    N = length(quats)
    w = √(1/N)
    Q = Matrix{Float64}(undef, 4, length(quats))
    @inbounds for (i, quat) in enumerate(quats)
        Q[1,i] = quat.w * w
        Q[2,i] = quat.x * w
        Q[3,i] = quat.y * w
        Q[4,i] = quat.z * w
    end
    return Q
end

display(x1)
display(x2)

Q = constructQ([x1, x2])
decompQ = eigen(Q*Q')
display(Q)
averagequat = UnitQuaternion(decompQ.vectors[:,4])
display(decompQ.vectors)
display(decompQ.values)
display(averagequat)


