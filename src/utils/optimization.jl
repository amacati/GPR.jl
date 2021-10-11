using BenchmarkTools

function optimize!(gpr::GaussianProcessRegressor)
    _optimize!(gpr, gpr.kernel)  # Dispatch depending on used kernel
    return gpr
end

function optimize!(mogpr::MOGaussianProcessRegressor)
    for gpr in mogpr
        _optimize!(gpr, gpr.kernel)  # Dispatch depending on used kernel
    end
    return mogpr
end

function _optimize!(gpr::GaussianProcessRegressor, kernel::GaussianKernel)
    lower = [1e-5, 1e-5]  # σ, λ both positive
    upper = [Inf, Inf]
    initial_params = [kernel.σ, kernel.λ]

    function fg!(F, G, x)
        try
            kernel = modifykernel(kernel, x[1], x[2])
            updategpr!(gpr, kernel)
            G !== nothing ? G[:] = gpr.parameter_gradient[:] : nothing
            F !== nothing && return - gpr.log_marginal_likelihood  # using negative to minimize instead of maximize log likelihood
        catch
            G !== nothing ? G[:] .= 0 : nothing
            F !== nothing && return 1e15  # Inf leads to problems  
        end
    end

    inner_optimizer = LBFGS()
    # inner_optimizer = GradientDescent()
    res = optimize(Optim.only_fg!(fg!), lower, upper, initial_params, Fminbox(inner_optimizer), Optim.Options(time_limit=10.))
    display(res)
    kernel = modifykernel(kernel, Optim.minimizer(res)...)
    updategpr!(gpr, kernel)
    return nothing
end

function _optimize!(gpr::GaussianProcessRegressor, kernel::GeneralGaussianKernel)
    S = length(kernel.Λ)
    N = length(gpr.Xstatic)
    lower = ones(S + 1) * 1e-5  # σ, Λ both positive
    upper = ones(S + 1) * 1e10
    initial_params = [kernel.σ, kernel.Λ...]
    ∂K∂θ = Matrix{Float64}(undef, N, N)
    K = Matrix{Float64}(undef, N, N)

    #= Gradients don't seem to work correctly
    function ∂K∂σ(kernel, K, α)
        for i in 1:N, j in 1:N
            ∂K∂σ!(kernel, gpr.Xstatic[i], gpr.Xstatic[j], ∂K∂θ, (i, j))
        end
        return 0.5*tr((α * α' - inv(K)) * ∂K∂θ)
    end

    function ∂K∂Λi(kernel, K, α, i)
        for j in 1:N, k in 1:N
            ∂K∂Λi!(kernel, gpr.Xstatic[j], gpr.Xstatic[k], ∂K∂θ, (j, k), i)
        end
        return 0.5*tr((α * α' - inv(K)) * ∂K∂θ)
    end

    function fg!(F, G, x)
        try
            kernel = GeneralGaussianKernel(x[1], x[2:end])
            L, α = Lα_decomposition!(gpr.Xstatic, gpr.Y, kernel, gpr.noisevariance, K)
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
        L, α = Lα_decomposition!(gpr.Xstatic, gpr.Y, kernel, gpr.noisevariance, K)
        return 0.5(gpr.Y*α)[1] + sum(log.(diag(L))) + size(gpr.Y,2)/2*log(2*pi)
    end


    # inner_optimizer = LBFGS()
    inner_optimizer = GradientDescent()
    # res = optimize(Optim.only_fg!(fg!), lower, upper, initial_params, Fminbox(inner_optimizer), Optim.Options(time_limit=10.))
    res = optimize(f, lower, upper, initial_params, Fminbox(inner_optimizer), Optim.Options(time_limit=10.))
    display(res)
    gpr.logPY = - Optim.minimum(res)  # undo negative log likelihood
    gpr.kernel = GeneralGaussianKernel(Optim.minimizer(res)[1], Optim.minimizer(res)[2:end])
    gpr.L, gpr.α = Lα_decomposition!(gpr.Xstatic, gpr.Y, gpr.kernel, gpr.noisevariance, K)
end
