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