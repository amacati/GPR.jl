function simplependulum2D(steps::Int; Δt::Real = 0.01, θstart::Real = 0., ωstart::Real = 0., Δm::Real = 1.0, ΔJ::Real = 1.0, friction::Real = 0, threadlock = nothing)
    joint_axis = [1.0; 0.0; 0.0]
    m = 1.0Δm
    g = -9.81
    l = 1.0
    r = 0.01
    p2 = [0.0;0.0;l / 2] # joint connection point
    ΔT = Δt * steps
    if threadlock === nothing
        mech, origin, link1 = _simplependulum2Dmech(r, l, m, ΔJ, joint_axis, p2, g, Δt)
    else
        lock(threadlock)  # ConstrainedDynamics uses globals with possible data races to initialize mechanisms -> lock
        try
            mech, origin, link1 = _simplependulum2Dmech(r, l, m, ΔJ, joint_axis, p2, g, Δt)
        finally
            unlock(threadlock)
        end
    end
    q1 = UnitQuaternion(RotX(θstart))
    setPosition!(origin, link1; p2=p2, Δq=q1)
    setVelocity!(origin, link1; p2=p2, Δω=SA[ωstart, 0., 0.])
    initialstates = [deepcopy(body.state) for body in mech.bodies]
    if friction != 0
        function control!(mech, _)
            setForce!(mech.bodies[1], τ=-SA[1.,0,0]friction*mech.bodies[1].state.ωc[1])
        end
        storage = simulate!(mech, ΔT, control!, record = true)
    else
        storage = simulate!(mech, ΔT, record = true)
    end    
    return storage, mech, initialstates
end

function _simplependulum2Dmech(r, l, m, ΔJ, joint_axis, p2, g, Δt)
    origin = Origin{Float64}()
    link1 = Cylinder(r, l, m)
    link1.J = I*1/12*m*l^2*ΔJ + zeros(3,3)  # Zeros to set broadcast shape
    joint_between_origin_and_link1 = EqualityConstraint(Revolute(origin, link1, joint_axis; p2=p2))
    links = [link1]
    constraints = [joint_between_origin_and_link1]
    return Mechanism(origin, links, constraints, g=g, Δt=Δt), origin, link1
end

function doublependulum2D(steps::Int; Δt::Real = 0.01, θstart::Vector{<:Real} = [0., 0.], ωstart::Vector{<:Real} = [0., 0.], Δm::Vector{<:Real} = ones(2),
                          ΔJ::Vector{<:Real} = ones(2), friction::Vector{<:Real} = zeros(2), threadlock = nothing)
    # Parameters
    l1 = 1.0
    l2 = 1.0
    m = ones(2) .* Δm
    x, y = .1, .1
    vert11 = [0.;0.;l1 / 2]
    vert12 = -vert11
    vert21 = [0.;0.;l2 / 2]
    joint_axis = [1.0; 0.0; 0.0]
    ΔT = Δt * steps
    # Initial orientation
    q1 = UnitQuaternion(RotX(θstart[1]))
    q2 = UnitQuaternion(RotX(θstart[2]))
    if threadlock === nothing
        mech, origin, link1, link2 = _doublependulum2Dmech(x, y, l1, l2, m, ΔJ, joint_axis, vert11, vert12, vert21, Δt)
    else
        lock(threadlock)
        try
            mech, origin, link1, link2 = _doublependulum2Dmech(x, y, l1, l2, m, ΔJ, joint_axis, vert11, vert12, vert21, Δt)
        finally
            unlock(threadlock)
        end
    end
    setPosition!(origin,link1, p2=vert11, Δq=q1)
    setPosition!(link1, link2, p1=vert12, p2=vert21, Δq=q2)
    setVelocity!(origin, link1; p2=vert11, Δω=SA[ωstart[1], 0., 0.])
    setVelocity!(link1, link2; p1=vert12, p2=vert21, Δω=SA[ωstart[2], 0., 0.])
    initialstates = [deepcopy(body.state) for body in mech.bodies]
    if (any(friction .!= 0))
        function control!(mech, _)
            setForce!(mech.bodies[1], τ=-SA[1.,0,0]friction[1]*mech.bodies[1].state.ωc[1])
            setForce!(mech.bodies[2], τ=-SA[1.,0,0]friction[2]*mech.bodies[2].state.ωc[1])
        end
        storage = simulate!(mech, ΔT, control!, record = true)
    else
        storage = simulate!(mech, ΔT, record = true)
    end
    return storage, mech, initialstates
end

function _doublependulum2Dmech(x, y, l1, l2, m, ΔJ, joint_axis, vert11, vert12, vert21, Δt)
    origin = Origin{Float64}()
    link1 = Box(x, y, l1, m[1], color = RGBA(1., 1., 0.))
    link2 = Box(x, y, l2, m[2], color = RGBA(1., 1., 0.))
    link1.J = I*1/12*m[1]*l1^2*ΔJ[1] + zeros(3,3)  # Zeros to set broadcast shape
    link2.J = I*1/12*m[2]*l2^2*ΔJ[2] + zeros(3,3)
    socket0to1 = EqualityConstraint(Revolute(origin, link1, joint_axis; p2=vert11))
    socket1to2 = EqualityConstraint(Revolute(link1, link2, joint_axis; p1=vert12, p2=vert21))
    links = [link1;link2]
    constraints = [socket0to1;socket1to2]
    mech = Mechanism(origin, links, constraints, Δt=Δt)
    return mech, origin, link1, link2
end

function cartpole(steps::Int; Δt::Real = 0.01, xstart::Real = 0., θstart::Real = 0., vstart::Real = 0., ωstart::Real = 0., 
                  Δm::Vector{<:Real} = ones(2), ΔJ::Real = 1., friction::Vector{<:Real} = zeros(2), threadlock = nothing)
    xaxis = [1.0; 0.0; 0.0]
    yaxis = [0.0; 1.0; 0.0]
    g = -9.81
    l = 0.5
    r = 0.01
    m = ones(2) .* Δm
    p01 = [0.0; 0.0; 0.0] # joint connection point
    p12 = [0.0; 0.0; 0.0]
    p21 = [0.0; 0.0; l/2]
    ΔT = steps*Δt
    x, y, z = 0.2, 0.3, 0.1
    if threadlock === nothing
        mech, origin, link1, link2 = _cartpolemech(x, y, z, r, l, m, ΔJ, xaxis, yaxis, p01, p12, p21, Δt)
    else
        lock(threadlock)
        try
            mech, origin, link1, link2 = _cartpolemech(x, y, z, r, l, m, ΔJ, xaxis, yaxis, p01, p12, p21, Δt)     
        finally
            unlock(threadlock)
        end
    end
    setPosition!(origin, link1; p2=p01, Δx=SA[0., xstart, 0.])
    setPosition!(link1, link2; p1=p12, p2=p21, Δq=UnitQuaternion(RotX(θstart)))
    setVelocity!(origin, link1; p2=p01, Δv=SA[0., vstart, 0.])
    setVelocity!(link1, link2; p1=p12, p2=p21, Δω=SA[ωstart, 0., 0.])
    initialstates = [deepcopy(body.state) for body in mech.bodies]
    if (any(friction .!= 0))
        function control!(mech, _)
            setForce!(mech.bodies[1], F=-SA[0,1.,0]friction[1]*mech.bodies[1].state.vc[2])
            setForce!(mech.bodies[2], τ=-SA[1.,0,0]friction[2]*mech.bodies[2].state.ωc[1])
        end
        storage = simulate!(mech, ΔT, control!, record = true)
    else
        storage = simulate!(mech, ΔT, record = true)
    end
    return storage, mech, initialstates
end

function _cartpolemech(x, y, z, r, l, m, ΔJ, xaxis, yaxis, p01, p12, p21, Δt)
    origin = Origin{Float64}()
    link1 = Box(x, y, z, m[1])
    link2 = Cylinder(r, l, m[2])
    # link1 inertia doesn't matter, no rotation
    link2.J = I*1/12*m[2]*l^2*ΔJ + zeros(3,3)  # Only xx matters
    joint_origin_link1 = EqualityConstraint(Prismatic(origin, link1, yaxis; p2=p01))
    joint_link1_link2 = EqualityConstraint(Revolute(link1, link2, xaxis; p1=p12, p2=p21))
    links = [link1, link2]
    constraints = [joint_origin_link1, joint_link1_link2]
    mech = Mechanism(origin, links, constraints, Δt=Δt)
    return mech, origin, link1, link2
end

function fourbar(steps::Int; Δt::Real = 0.01, θstart::Vector{<:Real} = [0., 0.], Δm::Vector{<:Real} = ones(4), ΔJ::Vector{<:Real} = ones(4), 
                 friction::Vector{<:Real} = zeros(4), threadlock = nothing)
    # Parameters
    ex = [1.;0.;0.]
    l = 1.0
    m = ones(4) .* Δm
    x, y = .1, .1
    vert11 = [0.;0.;l / 2]
    vert12 = -vert11
    ΔT = steps*Δt
    # Initial orientation
    q = UnitQuaternion(RotX(θstart[2]))
    qoff = UnitQuaternion(RotX(θstart[1]))
    Δq1 = q * qoff
    Δq2 = q

    if threadlock === nothing
        mech, origin, links = _fourbarmech(x, y, l, m, ΔJ, ex, vert11, vert12, Δt)
    else
        lock(threadlock)
        try
            mech, origin, links = _fourbarmech(x, y, l, m, ΔJ, ex, vert11, vert12, Δt)
        finally
            unlock(threadlock)
        end
    end
    setPosition!(origin, links[1], p2 = vert11, Δq = Δq1)
    setPosition!(links[1], links[2], p1 = vert12, p2 = vert11, Δq = inv(Δq2) * inv(Δq2))
    setPosition!(links[1], links[3], p1 = vert11, p2 = vert11, Δq = inv(Δq2) * inv(Δq2))
    setPosition!(links[3], links[4], p1 = vert12, p2 = vert11, Δq = Δq2 * Δq2)
    initialstates = [deepcopy(body.state) for body in mech.bodies]
    if (any(friction .!= 0))
        control!(mech, _) = for i in 1:4 setForce!(mech.bodies[i], τ=-SA[1.,0,0]friction[i]*mech.bodies[i].state.ωc[1]) end
        storage = simulate!(mech, ΔT, control!, record = true)
    else
        storage = simulate!(mech, ΔT, record = true)
    end
    return storage, mech, initialstates
end

function _fourbarmech(x, y, l, m, ΔJ, ex, vert11, vert12, Δt)
    origin = Origin{Float64}()
    links = [Box(x, y, l, m[i], color = RGBA(1., 1., 0.)) for i in 1:4]
    for (i, link) in enumerate(links)
        link.J = I*1/12*m[i]*l^2*ΔJ[i] + zeros(3,3)  # Zeros to set broadcast shape
    end
    j1 = EqualityConstraint(Revolute(origin, links[1], ex; p2=vert11))
    j2 = EqualityConstraint(Revolute(links[1], links[2], ex; p1=vert12, p2=vert11), Cylindrical(links[1], links[3], ex; p1=vert11, p2=vert11))
    j3 = EqualityConstraint(Revolute(links[3], links[4], ex; p1=vert12, p2=vert11))
    j4 = EqualityConstraint(Revolute(links[2], links[4], ex; p1=vert12, p2=vert12))
    constraints = [j1, j2, j3, j4]
    mech = Mechanism(origin, links, constraints, Δt=Δt)
    return mech, origin, links
end
