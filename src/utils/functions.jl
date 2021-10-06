# Faster version of tr(x1*x2) by only calculating diagonals 
function product_trace(x1::AbstractMatrix, x2::AbstractMatrix, temp::AbstractMatrix)
    temp .= x1 .* x2
    return sum(temp)
end

function quaternion_projection(q::UnitQuaternion)
    q.w >= 0 ? (return q) : return UnitQuaternion(-q.w, [-q.x, -q.y, -q.z])
end

function quaternion_to_array(q::UnitQuaternion)
    return SVector(q.w, q.x, q.y, q.z)
end

function quaternion_average(qvector::Vector{<:UnitQuaternion}, weights::Vector, mean::UnitQuaternion)
    N = length(qvector)
    @assert length(weights) == N
    meanweight = sqrt(max(1 - sum(weights),0))
    weights = sqrt.(weights)
    M = Matrix{Float64}(undef, 4, N+1)
    for i in 1:N
        M[:,i] = quaternion_to_array(quaternion_projection(qvector[i])) * weights[i]
    end
    M[:,end] = quaternion_to_array(quaternion_projection(mean)) * meanweight
    eigenvec = eigen(M*M').vectors[:,4]
    return UnitQuaternion(eigenvec[1], eigenvec[2:end])
end

function quaternion_average(qvector::Vector{<:UnitQuaternion}, weights::Vector)
    N = length(qvector)
    @assert length(weights) == N
    weights = sqrt.(weights)
    M = Matrix{Float64}(undef, 4, N)
    for i in 1:N
        M[:,i] = quaternion_to_array(quaternion_projection(qvector[i])) * weights[i]
    end
    eigenvec = eigen(M*M').vectors[:,4]
    return UnitQuaternion(eigenvec[1], eigenvec[2:end])
end


function quaternion_average(qvector::Vector{<:UnitQuaternion})
    N = length(qvector)
    M = Matrix{Float64}(undef, 4, N)
    for i in 1:N
        M[:,i] = quaternion_to_array(quaternion_projection(qvector[i]))
    end
    eigenvec = eigen(M*M').vectors[:,4]
    return UnitQuaternion(eigenvec[1], eigenvec[2:end])
end
