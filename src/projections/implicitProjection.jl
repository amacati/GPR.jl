using LinearAlgebra
using ConstrainedDynamics

function constraintmatrix(mechanism, eqc, body)
    x = ConstrainedDynamics.∂g∂ʳpos(mechanism, eqc, body.id)
    return x
end

function constructF(G)
    N1, N2 = size(G)
    F = zeros(N1+N2, N1+N2)
    F[1:N2, 1:N2] = Matrix(1.0I, N2, N2)
    F[1:N2, N2+1:N2+N1] = G'
    F[N2+1:end,1:N2] = G
    return F
end

function projectv!(v, newtonIter = 100)
    G = constraintmatrix()
    F = constructF(G)
    f(s, G) = v - s[1:3] + G's[4:end]
    s = SVector(v..., zeros(size(G,1))...)  # Initial s is [v, λ] with v = v, λ = 0
    for _ in 1:newtonIter
        G = constraintmatrix()
        Δs = F\f(s, G)
        s -= Δs
    end
    return s[1:3]
end


display(constraintmatrix(mechanism, eqc, body))