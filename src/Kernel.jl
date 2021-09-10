using LinearAlgebra

struct GaussianKernel

    variance::Number
    length::Number

end

function compute(kernel::GaussianKernel, x1::AbstractArray, x2::AbstractArray)::Number
    return kernel.variance^2 * exp(-norm(x1 - x2)^2/2kernel.length^2)
end