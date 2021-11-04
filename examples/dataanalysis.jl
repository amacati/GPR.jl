using Plots

include("utils.jl")


root = dirname(@__FILE__)

EXPERIMENT_IDs = Vector{String}()
for etype in ["CP_MAX", "P1_MAX", "P2_MAX"], nsamples in [2^i for i in 1:9]
    push!(EXPERIMENT_IDs, etype*string(nsamples)*"_FINAL")
end

results = Dict(id => loadcheckpoint(joinpath(root, "data", id)) for id in EXPERIMENT_IDs)

cp_error = Vector{Float64}()
for nsamples in [2^i for i in 1:9]
    onestep_msevec = results["CP_MAX"*string(nsamples)*"_FINAL"]["onestep_msevec"]
    filter!(x -> x!==nothing, onestep_msevec)
    min_mse = minimum(onestep_msevec)
    push!(cp_error, min_mse)
end

xvalues = [2^i for i in 1:9]
plt = bar(1:length(xvalues), cp_error, xticks = (1:length(xvalues), xvalues),
          xlabel = "Training samples",
          ylabel = "One Step Ahead forecast error", 
          title = "Cartpole maximal coordinates OSA error")
png(plt, "cp_error_max")

p1_error = Vector{Float64}()
for nsamples in [2^i for i in 1:9]
    onestep_msevec = results["P1_MAX"*string(nsamples)*"_FINAL"]["onestep_msevec"]
    filter!(x->x!==nothing, onestep_msevec)
    min_mse = minimum(onestep_msevec)
    push!(p1_error, min_mse)
end

plt = bar(1:length(xvalues), p1_error, xticks = (1:length(xvalues), xvalues),
          xlabel = "Training samples",
          ylabel = "One Step Ahead forecast error", 
          title = "1 link pendulum maximal coordinates OSA error")
png(plt, "p1_error_max")

p2_error = Vector{Float64}()
for nsamples in [2^i for i in 1:9]
    onestep_msevec = results["P2_MAX"*string(nsamples)*"_FINAL"]["onestep_msevec"]
    filter!(x->x!==nothing, onestep_msevec)
    min_mse = minimum(onestep_msevec)
    push!(p2_error, min_mse)
end

plt = bar(1:length(xvalues), p2_error, xticks = (1:length(xvalues), xvalues),
          xlabel = "Training samples",
          ylabel = "One Step Ahead forecast error", 
          title = "2 link pendulum maximal coordinates OSA error")
png(plt, "p2_error_max")
