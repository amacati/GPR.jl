struct CompositeKernel<:AbstractKernel
    kernels::Vector{<:AbstractKernel}
    kerneldims::Vector{<:Integer}
    nkernels::Integer

    function CompositeKernel(kernels::Vector{<:AbstractKernel}, kerneldims::Vector{<:Integer})
        new(kernels, kerneldims, length(kernels))
    end
end

Base.iterate(kernel::CompositeKernel, state=1) = state > kernel.nkernels ? nothing : (kernel.kernels[state], state+1)

@inline function compute(compositekernel::CompositeKernel, x1::AbstractArray, x2::AbstractArray)
    N = 1  # Each kernel is responsible for processing kernel.N dimensions of the state
    cov = 1
    @views for (id, kernel) in enumerate(compositekernel)
        cov *= compute(kernel, x1[N:N+compositekernel.kerneldims[id]-1], x2[N:N+compositekernel.kerneldims[id]-1])
        N += compositekernel.kerneldims[id]
    end
    return cov
end

@inline function compute!(compositekernel::CompositeKernel, x1::AbstractArray, x2::AbstractArray, target::AbstractMatrix, idx::Tuple{Integer, Integer})
    N = 1  # Each kernel is responsible for processing kernel.N dimensions of the state
    target[idx...] = 1
    @views for (id, kernel) in enumerate(compositekernel)
        target[idx...] *= compute(kernel, x1[N:N+compositekernel.kerneldims[id]-1], x2[N:N+compositekernel.kerneldims[id]-1])
        N += compositekernel.kerneldims[id]
    end
end
