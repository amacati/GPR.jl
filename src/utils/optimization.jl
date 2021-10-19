using BenchmarkTools

function optimize!(gpr::GaussianProcessRegressor; verbose::Bool = false)
    kernel = gpr.kernel
    lower = ones(kernel.Nparams) * 1e-5
    upper = ones(kernel.Nparams) * Inf
    initialparams = getinitialparams(kernel)
    function fg!(F, G, x)
        try
            modifykernel!(kernel, x)
            updategpr!(gpr, kernel)
            G !== nothing ? G[:] = gpr.parameter_gradient[:] : nothing
            F !== nothing && return - gpr.log_marginal_likelihood  # using negative to minimize instead of maximize log likelihood
        catch
            G !== nothing ? G[:] .= 0 : nothing
            F !== nothing && return 1e15  # Inf leads to problems  
        end
    end

    inner_optimizer = LBFGS()
    res = optimize(Optim.only_fg!(fg!), lower, upper, initialparams, Fminbox(inner_optimizer), Optim.Options(time_limit=10.))
    verbose && display(res)
    modifykernel!(kernel, Optim.minimizer(res))
    updategpr!(gpr, kernel)
    return gpr
end

function optimize!(mogpr::MOGaussianProcessRegressor)
    for gpr in mogpr
        _optimize!(gpr, gpr.kernel)  # Dispatch depending on used kernel
    end
    return mogpr
end
