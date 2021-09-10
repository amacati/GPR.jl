using Plots

function plot_gp(x, μ, σ)
    # Ribbon 2*std, but on both sides of the mean -> 4*σ
    plot(x, μ, ribbon = 4σ, fillalpha = 0.35, lw = 2, legend = :topleft,
         lab = "Gaussian Process Regression with 95% confidence interval")
end
