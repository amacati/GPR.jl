function optimize(kernel::GaussianKernel, X::Vector{SVector{S, Float64}}, Y::Matrix{Float64}, noisevariance::Float64)
    if noisevariance == 0
        # dont include noisevariance in optimization
    else
        # include noisevariance in optimization
    end

    return kernel, noisevariance
end