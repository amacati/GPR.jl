function max2mincoordinates(cstate::CState, mechanism::Mechanism)
    oldstates = getStates(mechanism)
    states = toStates(cstate)
    resetMechanism!(mechanism, states)
    cstate_min = Vector{Float64}()
    for eqc in mechanism.eqconstraints
        append!(cstate_min, ConstrainedDynamics.minimalCoordinates(mechanism, eqc))
        append!(cstate_min, ConstrainedDynamics.minimalVelocities(mechanism, eqc))
    end
    for (id, state) in enumerate(oldstates)
        mechanism.bodies[id].state = state  # Reset mechanism to default values
    end
    return cstate_min
end

"""
    Fourbar doesn't work well with the generic max2mincoordinates.
"""
function max2mincoordinates_fb(cstate::CState{T,4}) where T
    minstate = Vector{T}(undef, 4)
    q1 = UnitQuaternion(cstate[4], cstate[5:7])
    q2 = UnitQuaternion(cstate[30], cstate[31:33])  # angle 2 is body 3
    minstate[1] = Rotations.rotation_angle(q1)*sign(q1.x)*sign(q1.w)
    minstate[2] = cstate[11]
    minstate[3] = Rotations.rotation_angle(q2)*sign(q2.x)*sign(q2.w)
    minstate[4] = cstate[37]
    return minstate
end

"""
    Noise for v is calculated by doing a forward integration of ω in minimal coordinates, finding x2 and building a v that is consistent
    with the variational integration approach.
"""
function applynoise!(df, Σ, etype, Δtsim, varargs...)
    etype == "P1" && return _applynoise_p1!(df, Σ, Δtsim, varargs...)
    etype == "P2" && return _applynoise_p2!(df, Σ, Δtsim, varargs...)
    etype == "CP" && return _applynoise_cp!(df, Σ, Δtsim, varargs...)
    etype == "FB" && return _applynoise_fb!(df, Σ, Δtsim, varargs...)
    throw(ArgumentError("Experiment $etype is not supported!"))
end

function _applynoise_p1!(df, Σ, Δtsim, l)
    for col in eachcol(df)
        for t in 1:length(col)
            col[t][1].qc = UnitQuaternion(RotX(Σ["q"]*randn())) * col[t][1].qc  # Small error around θ
            col[t][1].ωc += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            θ = Rotations.rotation_angle(col[t][1].qc)*sign(col[t][1].qc.x)*sign(col[t][1].qc.w)  # Signum for axis direction
            ω = col[t][1].ωc[1]
            col[t][1].xc = [0, l/2*sin(θ), -l/2*cos(θ)]  # Noise is consequence of θ and ω
            θnext = ω*Δtsim + θ
            xnext = [0, l/2*sin(θnext), -l/2*cos(θnext)]
            v = (xnext - col[t][1].xc)/Δtsim
            col[t][1].vc = v
        end
    end
end

function _applynoise_p2!(df, Σ, Δtsim, l1, l2)
    for col in eachcol(df)
        for t in 1:length(col)
            col[t][1].qc = UnitQuaternion(RotX(Σ["q"]*randn())) * col[t][1].qc
            col[t][1].ωc += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            col[t][2].qc = UnitQuaternion(RotX(Σ["q"]*randn())) * col[t][2].qc
            col[t][2].ωc += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            θ1 = Rotations.rotation_angle(col[t][1].qc)*sign(col[t][1].qc.x)*sign(col[t][1].qc.w)  # Signum for axis direction
            θ2 = Rotations.rotation_angle(col[t][2].qc)*sign(col[t][2].qc.x)*sign(col[t][2].qc.w) - θ1
            ω1, ω2 = col[t][1].ωc[1], col[t][2].ωc[1] - col[t][1].ωc[1]
            col[t][1].xc = [0, l1/2*sin(θ1), -l1/2*cos(θ1)]  # Noise is consequence of θ and ω
            col[t][2].xc = [0, l1*sin(θ1) + l2/2*sin(θ1+θ2), -l1*cos(θ1) - l2/2*cos(θ1+θ2)]  # Noise is consequence of θ and ω
            θ1next = θ1 + ω1*Δtsim
            θ2next = θ2 + ω2*Δtsim
            x1next = [0, l1/2*sin(θ1next), -l1/2*cos(θ1next)]
            x2next = [0, l1*sin(θ1next) + l2/2*sin(θ1next+θ2next), -l1*cos(θ1next) - l2/2*cos(θ1next+θ2next)]
            v1 = (x1next - col[t][1].xc) / Δtsim
            v2 = (x2next - col[t][2].xc) / Δtsim
            col[t][1].vc = v1
            col[t][2].vc = v2
        end
    end
end

function _applynoise_cp!(df, Σ, Δtsim, l)
    for col in eachcol(df)
        for t in 1:length(col)
            col[t][1].xc += Σ["x"]*[0, randn(), 0]  # Cart pos noise only y, no orientation noise
            col[t][1].vc += Σ["v"]*[0, randn(), 0]  # Same for v
            col[t][2].qc = UnitQuaternion(RotX(Σ["q"]*randn())) * col[t][2].qc
            col[t][2].ωc += Σ["ω"]*[randn(), 0, 0]
            θ = Rotations.rotation_angle(col[t][2].qc)*sign(col[t][2].qc.x)*sign(col[t][2].qc.w)  # Signum for axis direction
            ω = col[t][2].ωc[1]
            col[t][2].xc = col[t][1].xc + [0, l/2*sin(θ), -l/2*cos(θ)]
            θnext = ω*Δtsim + θ
            xnext = col[t][1].xc + col[t][1].vc*Δtsim + [0, l/2*sin(θnext), -l/2*cos(θnext)]
            v = (xnext - col[t][2].xc)/Δtsim
            col[t][2].vc = v
        end
    end
end

function _applynoise_fb!(df, Σ, Δtsim, l)
    for col in eachcol(df)
        for t in 1:length(col)
            col[t][1].qc = UnitQuaternion(RotX(Σ["q"]*randn())) * col[t][1].qc
            col[t][1].ωc += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            col[t][3].qc = UnitQuaternion(RotX(Σ["q"]*randn())) * col[t][3].qc
            col[t][3].ωc += Σ["ω"]*[randn(), 0, 0]  # Zero noise in fixed ωy, ωz
            θ1 = Rotations.rotation_angle(col[t][1].qc)*sign(col[t][1].qc.x)*sign(col[t][1].qc.w)  # Signum for axis direction
            θ2 = Rotations.rotation_angle(col[t][3].qc)*sign(col[t][3].qc.x)*sign(col[t][3].qc.w)
            ω1, ω2 = col[t][1].ωc[1], col[t][3].ωc[1]
            col[t][2].qc = UnitQuaternion(RotX(θ2))
            col[t][4].qc = UnitQuaternion(RotX(θ1))
            col[t][1].xc = [0, 0.5sin(θ1)l, -0.5cos(θ1)l]
            col[t][2].xc = [0, sin(θ1)l + 0.5sin(θ2)l, -cos(θ1)l - 0.5cos(θ2)l]
            col[t][3].xc = [0, 0.5sin(θ2)l, -0.5cos(θ2)l]
            col[t][4].xc = [0, sin(θ2)l + 0.5sin(θ1)l, -cos(θ2)l - 0.5cos(θ1)l]
            θ1next = ω1*Δtsim + θ1
            θ2next = ω2*Δtsim + θ2
            x1next = [0, 0.5sin(θ1next)l, -0.5cos(θ1next)l]
            x2next = [0, sin(θ1next)l + 0.5sin(θ2next)l, -cos(θ1next)l - 0.5cos(θ2next)l]
            x3next = [0, 0.5sin(θ2next)l, -0.5cos(θ2next)l]
            x4next = [0, sin(θ2next)l + 0.5sin(θ1next)l, -cos(θ2next)l - 0.5cos(θ1next)l]
            v1 = (x1next - col[t][1].xc)/Δtsim
            v2 = (x2next - col[t][2].xc)/Δtsim
            v3 = (x3next - col[t][3].xc)/Δtsim
            v4 = (x4next - col[t][4].xc)/Δtsim
            col[t][1].vc = v1
            col[t][2].vc = v2
            col[t][3].vc = v3
            col[t][4].vc = v4
            col[t][2].ωc = col[t][3].ωc
            col[t][4].ωc = col[t][1].ωc
        end
    end
end
