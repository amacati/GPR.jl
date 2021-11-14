using Plots
using Statistics
using StatsPlots
using Statistics
using CategoricalArrays

include("utils.jl")


root = dirname(@__FILE__)

checkpoint = loadcheckpoint("noisy_final")
EXPERIMENT_IDs = Vector{String}()
vnsamples = [2^i for i in 1:9]

function create_plot(kstep_median, kstep_std, abbrev)
    sx = repeat(["Max", "Min"], inner = length(vnsamples))
    nam = CategoricalArray(repeat([string(i) for i in vnsamples], outer = 2))
    levels!(nam, [string(i) for i in vnsamples])
    plt = groupedbar(nam, kstep_median, yerror = kstep_std, group = sx, 
                     title = "$abbrev OSA error", xlabel = "Training samples", 
                     ylabel = "K steps ahead forecast error")
    png(plt, "$(abbrev)_error")
end

# Maximal coordinates
# Cartpole
for id in ["CP", "P1", "P2"]# , "FB"]
    kstep_median = Vector{Float64}()
    kstep_std = Vector{Float64}()
    for nsamples in vnsamples
        kstep_mse = checkpoint[id*"_MAX"*string(nsamples)]["kstep_mse"]
        filter!(x -> x!==nothing, kstep_mse)
        push!(kstep_median, median(kstep_mse))
        push!(kstep_std, std(kstep_mse))
    end
    for nsamples in vnsamples
        kstep_mse = checkpoint[id*"_MIN"*string(nsamples)]["kstep_mse"]
        filter!(x -> x!==nothing, kstep_mse)
        push!(kstep_median, median(kstep_mse))
        push!(kstep_std, std(kstep_mse))
    end
    create_plot(kstep_median, kstep_std, id)
end