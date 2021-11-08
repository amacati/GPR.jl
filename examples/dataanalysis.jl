using Plots
using StatsPlots
using Statistics
using CategoricalArrays

include("utils.jl")


root = dirname(@__FILE__)

checkpoint = loadcheckpoint("noisy_final")
EXPERIMENT_IDs = Vector{String}()
vnsamples = [2^i for i in 1:9]

function create_plot(kstep_mean, kstep_std, etype, abbrev, coords)
    sx = repeat(["Max", "Min"], inner = length(vnsamples))

    nam = CategoricalArray(repeat([string(i) for i in vnsamples], outer = 2))
    levels!(nam, [string(i) for i in vnsamples])
    display(kstep_std)
    plt = groupedbar(nam, kstep_mean, yerr = kstep_std, group = sx, 
                     title = "$etype $coords coordinates OSA error", xlabel = "Training samples", 
                     ylabel = "K steps ahead forecast error")
    png(plt, "$(abbrev)_error_$coords")
end

# Maximal coordinates
# Cartpole
kstep_mean = Vector{Float64}()
kstep_std = Vector{Float64}()
for nsamples in vnsamples
    kstep_mse = checkpoint["CP_MAX"*string(nsamples)]["kstep_mse"]
    filter!(x -> x!==nothing, kstep_mse)
    push!(kstep_mean, mean(kstep_mse))
    push!(kstep_std, std(kstep_mse))
end
for nsamples in vnsamples
    kstep_mse = checkpoint["CP_MIN"*string(nsamples)]["kstep_mse"]
    filter!(x -> x!==nothing, kstep_mse)
    push!(kstep_mean, mean(kstep_mse))
    push!(kstep_std, std(kstep_mse))
end
create_plot(kstep_mean, kstep_std, "Cartpole", "cp", "maximal")

kstep_mean = Vector{Float64}()
kstep_std = Vector{Float64}()
for nsamples in vnsamples
    kstep_mse = checkpoint["P1_MAX"*string(nsamples)]["kstep_mse"]
    filter!(x -> x!==nothing, kstep_mse)
    push!(kstep_mean, mean(kstep_mse))
    push!(kstep_std, std(kstep_mse))
end
for nsamples in vnsamples
    kstep_mse = checkpoint["P1_MIN"*string(nsamples)]["kstep_mse"]
    filter!(x -> x!==nothing, kstep_mse)
    push!(kstep_mean, mean(kstep_mse))
    push!(kstep_std, std(kstep_mse))
end
create_plot(kstep_mean, kstep_std, "1 Link pendulum", "p1", "maximal")

kstep_mean = Vector{Float64}()
kstep_std = Vector{Float64}()
for nsamples in vnsamples
    kstep_mse = checkpoint["P2_MAX"*string(nsamples)]["kstep_mse"]
    filter!(x -> x!==nothing, kstep_mse)
    push!(kstep_mean, mean(kstep_mse))
    push!(kstep_std, std(kstep_mse))
end
for nsamples in vnsamples
    kstep_mse = checkpoint["P2_MIN"*string(nsamples)]["kstep_mse"]
    filter!(x -> x!==nothing, kstep_mse)
    push!(kstep_mean, mean(kstep_mse))
    push!(kstep_std, std(kstep_mse))
end
create_plot(kstep_mean, kstep_std, "2 link pendulum", "p2", "maximal")
