using ConstrainedDynamics
using ConstrainedDynamicsVis


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

T0 = 2*pi*sqrt((m*l^2/3)/(-g*l/2))
t0 = 1

q1 = UnitQuaternion(RotX(π / 2))
setPosition!(origin,link1,p2 = p2,Δq = q1)
setVelocity!(link1)
storage = simulate!(mech,10.,record = true)

display(storage.x[1][10:20])

ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")
