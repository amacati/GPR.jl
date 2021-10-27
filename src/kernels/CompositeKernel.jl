struct CompositeKernel<:AbstractKernel
    kernels::Vector{<:AbstractKernel}
    kerneldims::Vector{<:Integer}
    nkernels::Integer
    nparams::Integer

    function CompositeKernel(kernels::Vector{<:AbstractKernel}, kerneldims::Vector{<:Integer})
        @assert length(kernels) == length(kerneldims)  ("Number of kernel dimensions has to equal number of kernels!")
        nparams = sum([kernel.nparams for kernel in kernels])
        nkernels = length(kernels)
        new(kernels, kerneldims, nkernels, nparams)
    end
end

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

function _chained_derivative!(compositekernel, X, target, kernelid, ∂K∂θi)
    kernel = compositekernel.kernels[kernelid]
    N = sum(compositekernel.kerneldims[1:kernelid-1]) + 1
    ∂K∂θi(kernel, [@view i[N:N+compositekernel.kerneldims[kernelid]-1] for i in X], target)  # Overwrites the target with values of ∂K∂θi
    N = 1  # Each kernel is responsible for processing kernel.N dimensions of the state
    @views for (id, kernel) in enumerate(compositekernel)
        if id == kernelid
            N += compositekernel.kerneldims[id]  # Skip this dimension, already covered in the derivative handle
            continue
        end
        for i in 1:length(X), j in 1:i
            target[i, j] *= compute(kernel, X[i][N:N+compositekernel.kerneldims[id]-1], X[j][N:N+compositekernel.kerneldims[id]-1])
        end
        N += compositekernel.kerneldims[id]
    end
    return Symmetric(target, :L)
end

# Careful when operating on changing datasets! Cache does not check if the dataset has changed and can become stale!
function get_derivative_handles(compositekernel::CompositeKernel)
    derivative_handles = Vector{Function}(undef, compositekernel.nparams)
    ∂K∂θ = [[get_derivative_handles(kernel)[∂id] for ∂id in 1:kernel.nparams] for kernel in compositekernel]
    N = 1
    for (kernelid, kernel) in enumerate(compositekernel)
        derivative_handles[N:N+kernel.nparams-1] = [chained_derivative!(ckernel, X, target) = _chained_derivative!(ckernel, X, target, kernelid, ∂K∂θ[kernelid][∂id]) for ∂id in 1:kernel.nparams]
        N += kernel.nparams
    end
    return derivative_handles
end

# Modify kernel in place to avoid creating new kernel objects in optimizations.
function modifykernel!(compositekernel::CompositeKernel, param::AbstractArray)
    @assert length(param) == compositekernel.nparams ("param vector has wrong number of parameters!")
    N = 1
    for kernel in compositekernel
        modifykernel!(kernel, param[N:N+kernel.nparams-1])
        N += kernel.nparams
    end
end

function getparams(kernel::CompositeKernel)
    params = Vector{Float64}(undef, kernel.nparams)
    N = 1
    for k in kernel.kernels
        params[N:N+k.nparams-1] = getparams(k)
        N += k.nparams
    end
    return params
end

Base.iterate(kernel::CompositeKernel, state=1) = state > kernel.nkernels ? nothing : (kernel.kernels[state], state+1)

function Base.length(s::CompositeKernel)
    return s.nkernels
end

function Base.copy(s::CompositeKernel)
    return CompositeKernel([copy(kernel) for kernel in s], s.kerneldims)
end
