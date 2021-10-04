function optimize!(gpr::GaussianProcessRegressor)
    _optimize!(gpr, gpr.kernel)  # Dispatch depending on used kernel
end

function optimize!(mogpr::MOGaussianProcessRegressor)
    for gpr in mogpr
        _optimize!(gpr, gpr.kernel)  # Dispatch depending on used kernel
    end
end

# Faster version of tr(x1*x2) by only calculating diagonals 
function product_trace(x1::AbstractMatrix, x2::AbstractMatrix, temp::AbstractMatrix)
    temp .= x1 .* x2
    return sum(temp)
end

function _optimize!(gpr::GaussianProcessRegressor, kernel::GaussianKernel)
    N = length(gpr._X)
    lower = [1e-5, 1e-5]  # σ, λ both positive
    upper = [Inf, Inf]
    initial_params = [kernel.σ, kernel.λ]
    tracetemp = Matrix{Float64}(undef, N, N)
    ∂K∂θ = similar(tracetemp)
    K = similar(tracetemp)
    KinvY = Matrix{Float64}(undef, N, 1)
    ααT = similar(tracetemp)

    function ∂K∂σ(kernel, ααTK)
        for i in 1:N, j in 1:i
            ∂K∂σ!(kernel, gpr._X[i], gpr._X[j], ∂K∂θ, (i, j))
        end
        return - 0.5*product_trace(ααTK, Symmetric(∂K∂θ, :L), tracetemp)
    end

    function ∂K∂λ(kernel, ααTK)
        for i in 1:N, j in 1:i
            @inbounds ∂K∂λ!(kernel, gpr._X[i], gpr._X[j], ∂K∂θ, (i, j))
        end
        return - 0.5*product_trace(ααTK, Symmetric(∂K∂θ, :L), tracetemp)
    end

    function fg!(F, G, x)
        kernel = GaussianKernel(x...)
        L, α = Lα_decomposition!(gpr._X, gpr.Y, kernel, gpr.noisevariance, K)  # In place calculation of K, still need to use K with Symmetric(K, :L)!
        function compute_gradient()
            try
                Kinv = inv(cholesky(Symmetric(K, :L)))  # K is symmetric, cholesky should be faster
                mul!(KinvY, Kinv, gpr.Y')
                mul!(ααT, KinvY, KinvY')
                ααTK = ααT - Kinv  # ααTK = (α*α' - inv(K)) with α = inv(K)*y'
                G[1] = ∂K∂σ(kernel, ααTK)
                G[2] = ∂K∂λ(kernel, ααTK)
            catch  # In case cholesky fails
                G[1] = 0
                G[2] = 0
            end
        end
        if G !== nothing
            @time compute_gradient()
        end
        if F !== nothing
            # using negative to minimize instead of maximize log likelihood
            return 0.5(gpr.Y*α)[1] + sum(log.(diag(L))) + size(gpr.Y,2)/2*log(2*pi)
        end    
    end

    function f(x)
        kernel = GaussianKernel(x...)
        L, α = Lα_decomposition!(gpr._X, gpr.Y, kernel, gpr.noisevariance, K)
        return 0.5(gpr.Y*α)[1] + sum(log.(diag(L))) + size(gpr.Y,2)/2*log(2*pi)
    end

    # inner_optimizer = LBFGS()
    inner_optimizer = GradientDescent()
    res = optimize(Optim.only_fg!(fg!), lower, upper, initial_params, Fminbox(inner_optimizer), Optim.Options(time_limit=10.))
    # res = optimize(f, lower, upper, initial_params, Fminbox(inner_optimizer), Optim.Options(time_limit=10.))
    display(res)
    gpr.logPY = - Optim.minimum(res)  # undo negative log likelihood
    gpr.kernel = GaussianKernel(Optim.minimizer(res)...)
    gpr.L, gpr.α = Lα_decomposition!(gpr._X, gpr.Y, gpr.kernel, gpr.noisevariance, K)
end

function _optimize!(gpr::GaussianProcessRegressor, kernel::GeneralGaussianKernel)
    S = length(kernel.Λ)
    N = length(gpr._X)
    lower = ones(S + 1) * 1e-5  # σ, Λ both positive
    upper = ones(S + 1) * 1e10
    initial_params = [kernel.σ, kernel.Λ...]
    ∂K∂θ = Matrix{Float64}(undef, N, N)
    K = Matrix{Float64}(undef, N, N)

    #= Gradients don't seem to work correctly
    function ∂K∂σ(kernel, K, α)
        for i in 1:N, j in 1:N
            ∂K∂σ!(kernel, gpr._X[i], gpr._X[j], ∂K∂θ, (i, j))
        end
        return 0.5*tr((α * α' - inv(K)) * ∂K∂θ)
    end

    function ∂K∂Λi(kernel, K, α, i)
        for j in 1:N, k in 1:N
            ∂K∂Λi!(kernel, gpr._X[j], gpr._X[k], ∂K∂θ, (j, k), i)
        end
        return 0.5*tr((α * α' - inv(K)) * ∂K∂θ)
    end

    function fg!(F, G, x)
        try
            kernel = GeneralGaussianKernel(x[1], x[2:end])
            L, α = Lα_decomposition!(gpr._X, gpr.Y, kernel, gpr.noisevariance, K)
            if G !== nothing
                G[1] = ∂K∂σ(kernel, K, α)
                for i in 2:length(x)
                    G[i] = ∂K∂Λi(kernel, K, α, i-1)
                end
            end
            if F !== nothing
                # using negative to minimize instead of maximize log likelihood
                return 0.5(gpr.Y*α)[1] + sum(log.(diag(L))) + size(gpr.Y,2)/2*log(2*pi)
            end    
        catch e
            throw(e)  # TODO: REMOVE, DEBUG only_fg
            return 1e15  # Inf crashes some optimization algorithms
        end
    end
    =#
    function f(x)
        kernel = GeneralGaussianKernel(x[1], x[2:end])
        L, α = Lα_decomposition!(gpr._X, gpr.Y, kernel, gpr.noisevariance, K)
        return 0.5(gpr.Y*α)[1] + sum(log.(diag(L))) + size(gpr.Y,2)/2*log(2*pi)
    end


    # inner_optimizer = LBFGS()
    inner_optimizer = GradientDescent()
    # res = optimize(Optim.only_fg!(fg!), lower, upper, initial_params, Fminbox(inner_optimizer), Optim.Options(time_limit=10.))
    res = optimize(f, lower, upper, initial_params, Fminbox(inner_optimizer), Optim.Options(time_limit=10.))
    display(res)
    gpr.logPY = - Optim.minimum(res)  # undo negative log likelihood
    gpr.kernel = GeneralGaussianKernel(Optim.minimizer(res)[1], Optim.minimizer(res)[2:end])
    gpr.L, gpr.α = Lα_decomposition!(gpr._X, gpr.Y, gpr.kernel, gpr.noisevariance, K)
end
