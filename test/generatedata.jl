using ConstrainedDynamics
using ConstrainedDynamicsVis

function simple_pendulum()
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
    return storage
end

