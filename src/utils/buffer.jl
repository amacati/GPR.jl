mutable struct GPRGradientBuffer{N}

    Kinv::AbstractMatrix
    α::AbstractMatrix
    ααT::AbstractMatrix
    ααTK::AbstractMatrix
    ∂K∂θ::AbstractMatrix
    tracetmp::AbstractMatrix

    function GPRGradientBuffer{N}() where N
        Kinv = Matrix{Float64}(undef, N, N)
        α = Matrix{Float64}(undef, N, 1)
        ααT = similar(Kinv)
        ααTK = similar(Kinv)
        ∂K∂θ = similar(Kinv)
        tracetmp = similar(Kinv)
        new{N}(Kinv, α, ααT, ααTK, ∂K∂θ, tracetmp)
    end

end

function updatebuffer!(buffer::GPRGradientBuffer, Kinv::AbstractMatrix, Y::AbstractMatrix)
    buffer.Kinv = Kinv
    mul!(buffer.α, Kinv, Y')
    mul!(buffer.ααT, buffer.α, buffer.α')
    buffer.ααTK = buffer.ααT - Kinv
    return buffer
end