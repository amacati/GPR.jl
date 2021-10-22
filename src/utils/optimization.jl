using BenchmarkTools

function optimize!(gpr::GaussianProcessRegressor; verbose::Bool = false)
    kernel = gpr.kernel
    lower = ones(kernel.nparams) * 1e-5
    upper = ones(kernel.nparams) * Inf
    initialparams = getparams(kernel)
    function fg!(F, G, x)
        try
            modifykernel!(kernel, x)
            updategpr!(gpr, kernel)
            G !== nothing ? G[:] = gpr.parametergradient[:] : nothing
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

function optimize!(mogpr::MOGaussianProcessRegressor; verbose::Bool = False, threading::Bool = true)
    if threading
        Threads.@threads for gpr in mogpr
            optimize!(gpr; verbose=verbose)
        end
    else
        for gpr in mogpr
            optimize!(gpr; verbose=verbose)
        end
    end
    return mogpr
end
