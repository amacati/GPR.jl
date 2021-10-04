function plot_gp(x, μ, σ)
    # Plotting needs reshaping to column vector
    x = size(x,1) == 1 && size(x,2) != 1 ? reshape(x, :, 1) : x
    # Ribbon 2*std, but on both sides of the mean -> 4*σ
    plot(x, μ, ribbon = 4σ, fillalpha = 0.35, lw = 2, legend = :topleft,
         lab = "Gaussian Process Regression with 95% confidence interval")
end

function plot_gp(xlim::Tuple{Real, Real}, ylim::Tuple{Real, Real}, f::Function)
    pyplot()
    plot3d(range(xlim..., length=100), range(ylim..., length=100), f, st=:surface, camera=(-30,30),
                 xlabel = "Feature 1", ylabel = "Feature 2", zlabel = "GP interpolation")
end

function plot_gp(xlim::Tuple{Real, Real}, ylim::Tuple{Real, Real}, f::Function, ftrue::Function)
    pyplot()
    plot(plot3d(range(xlim..., length=100), range(ylim..., length=100), f, st=:surface, camera=(-30,30),
                xlabel = "Feature 1", ylabel = "Feature 2", zlabel = "GP interpolation"),
         plot3d(range(xlim..., length=100), range(ylim..., length=100), ftrue, st=:surface, camera=(-30,30),
                xlabel = "Feature 1", ylabel = "Feature 2", zlabel = "Ground truth"))
end

function visualize_prediction(mech, X::AbstractArray{<:AbstractArray{<:AbstractArray{<:Real}}})
    N = length(X[1])
    storage = ConstrainedDynamicsVis.Storage{Float64}(Base.OneTo(N),length(X))
    for id in 1:length(X)
        for t in 1:N
            state = X[id][t]
            storage.x[id][t] = state[1:3]
            storage.q[id][t] = UnitQuaternion(state[4], state[5:7])
        end
    end
    ConstrainedDynamicsVis.visualize(mech, storage; showframes = true, env = "editor")
end
