using ConstrainedDynamics
using ConstrainedDynamics: GenericJoint,Vmat,params
using ConstrainedDynamicsVis
using StaticArrays

function load2Ddata(storage::Storage, coord="max")
    coord == "max" && return _load2Ddatamax(storage)
    coord == "min" && return _load2Ddatamin(storage)
    throw(ArgumentError("Coordinate argument \"$coord\" not supported!"))
end

function load3Ddata(storage::Storage, coord="max")
    coord == "max" && return _load3Ddatamax(storage)
    coord == "min" && return _load3Ddatamin(storage)
    throw(ArgumentError("Coordinate argument \"$coord\" not supported!"))
end

function _load2Ddatamax(storage)
    S = length(storage.x)
    N = length(storage.x[1])
    x = Vector{Matrix{Float64}}(undef, S)
    for body_id in 1:S
        x[S] = Matrix{Float64}(undef, 6, N)  # 2 pos 1 quat 2 linearV 1 angularV
        for t in 1:N
            x[S][1:2,t] = storage.x[body_id][t][2:3]  # Y and Z act as X and Y
            quat = storage.q[body_id][t]
            x[S][3,t] = sin(RotXYZ(quat).theta1)  # Sin(θ) acts as quaternion in 2D
            x[S][4:5,t] = storage.v[body_id][t][2:3]
            x[S][6,t] = storage.ω[body_id][t][1]
        end
    end
    return x
end

function _load2Ddatamin(storage)
    S = length(storage.x)
    N = length(storage.x[1])
    x = Vector{Matrix{Float64}}(undef, S)
    for body_id in 1:S
        x[S] = Matrix{Float64}(undef, 2, N)  # 1 pos 1 velocity
        for t in 1:N
            quat = storage.q[body_id][t]
            x[S][1,t] = RotXYZ(quat).theta1
            x[S][2,t] = storage.ω[body_id][t][1]
        end
    end
    return x
end

function _load3Ddatamax(storage)
    S = length(storage.x)
    N = length(storage.x[1])
    x = Vector{Matrix{Float64}}(undef, S)
    for body_id in 1:S
        x[S] = Matrix{Float64}(undef, 13, N)  # 3 pos 4 quat 3 linearV 3 angularV
        for t in 1:N
            x[S][1:3,t] = storage.x[body_id][t]
            quat = storage.q[body_id][t]
            x[S][4:7,t] = [quat.w, quat.x, quat.y, quat.z]
            x[S][8:10,t] = storage.v[body_id][t]
            x[S][11:13] = storage.ω[body_id][t]
        end
    end
    return x
end

function _load3Ddatamin(storage)
    S = length(storage.x)
    N = length(storage.x[1])
    x = Vector{Matrix{Float64}}(undef, S)
    for body_id in 1:S
        x[S] = Matrix{Float64}(undef, 6, N)  # 3 pos 3 angularV
        for t in 1:N
            euler = RotXYZ(storage.q[body_id][t])
            x[S][1:3,t] = [euler.theta1, euler.theta2, euler.theta3]
            x[S][4:6,t] = storage.ω[body_id][t]
        end
    end
    return x
end

function simplependulum2D()
    joint_axis = [1.0; 0.0; 0.0]

    Δt=0.01
    g = -9.81
    m = 1.0
    l = 1.0
    r = 0.01

    p2 = [0.0;0.0;l / 2] # joint connection point

    # Links
    origin = Origin{Float64}()
    link1 = Cylinder(r, l, m)

    # Constraints
    joint_between_origin_and_link1 = EqualityConstraint(Revolute(origin, link1, joint_axis; p2=p2))

    links = [link1]
    constraints = [joint_between_origin_and_link1]


    mech = Mechanism(origin, links, constraints, g=g,Δt=Δt)

    q1 = UnitQuaternion(RotX(π / 2))
    setPosition!(origin,link1,p2 = p2,Δq = q1)
    setVelocity!(link1)
    storage = simulate!(mech,10.,record = true)
    return storage, mech
end

function doublependulum3D()
    # Parameters
    l1 = 1.0
    l2 = sqrt(2) / 2
    x, y = .1, .1
    vert11 = [0.;0.;l1 / 2]
    vert12 = -vert11
    vert21 = [0.;0.;l2 / 2]

    # Initial orientation
    phi1 = pi / 4
    q1 = UnitQuaternion(RotX(phi1))

    # Links
    origin = Origin{Float64}()
    link1 = Box(x, y, l1, l1, color = RGBA(1., 1., 0.))
    link2 = Box(x, y, l2, l2, color = RGBA(1., 1., 0.))

    # Constraints
    socket0to1 = EqualityConstraint(Spherical(origin, link1; p2=vert11))
    socket1to2 = EqualityConstraint(Spherical(link1, link2; p1=vert12, p2=vert21))

    links = [link1;link2]
    constraints = [socket0to1;socket1to2]
    mech = Mechanism(origin, links, constraints)
    setPosition!(origin,link1,p2 = vert11,Δq = q1)
    setPosition!(link1,link2,p1 = vert12,p2 = vert21,Δq = inv(q1)*UnitQuaternion(RotY(0.2)))

    storage = simulate!(mech, 10., record = true)
    return storage, mech
end

function simplependulum3D()
    # Parameters
    l1 = 1.0
    x, y = .1, .1
    vert11 = [0.;0.;l1 / 2]

    # Initial orientation
    phi1 = pi / 4
    q1 = UnitQuaternion(RotX(phi1))

    # Links
    origin = Origin{Float64}()
    link1 = Box(x, y, l1, l1, color = RGBA(1., 1., 0.))
    # link2 = Box(x, y, l2, l2, color = RGBA(1., 1., 0.))

    # Constraints
    socket0to1 = EqualityConstraint(Spherical(origin, link1; p2=vert11))
    
    links = [link1]
    constraints = [socket0to1]

    mech = Mechanism(origin, links, constraints)
    setPosition!(origin,link1,p2 = vert11,Δq = q1)
    setVelocity!(link1, v=[1., 0., 0])

    storage = simulate!(mech, 10., record = true)
    return storage, mech
end