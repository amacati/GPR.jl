using Plots

include("utils.jl")


root = dirname(@__FILE__)

EXPERIMENT_IDs = Vector{String}()
vnsamples = [2^i for i in 4:9]
for etype in ["CP_MAX", "P1_MAX", "P2_MAX", "CP_MIN", "P1_MIN", "P2_MIN"], nsamples in vnsamples
    push!(EXPERIMENT_IDs, etype*string(nsamples)*"_FINAL")
end

results = Dict(id => loadcheckpoint(id) for id in EXPERIMENT_IDs)

function create_osa_plot(xvalues, error, etype, abbrev, coords)
    plt = bar(1:length(xvalues), error, xticks = (1:length(xvalues), xvalues),
    xlabel = "Training samples",
    ylabel = "One Step Ahead forecast error", 
    title = "$etype $coords coordinates OSA error")
    png(plt, "$(abbrev)_error_$coords")
end

# Maximal coordinates
# Cartpole
error = Vector{Float64}()
for nsamples in vnsamples
    onestep_msevec = results["CP_MAX"*string(nsamples)*"_FINAL"]["onestep_msevec"]
    filter!(x -> x!==nothing, onestep_msevec)
    min_mse = minimum(onestep_msevec)
    push!(error, min_mse)
end
create_osa_plot(vnsamples, error, "Cartpole", "cp", "maximal")

error = Vector{Float64}()
for nsamples in vnsamples
    onestep_msevec = results["P1_MAX"*string(nsamples)*"_FINAL"]["onestep_msevec"]
    filter!(x->x!==nothing, onestep_msevec)
    min_mse = minimum(onestep_msevec)
    push!(error, min_mse)
end
create_osa_plot(vnsamples, error, "1 link pendulum", "p1", "maximal")


error = Vector{Float64}()
for nsamples in vnsamples
    onestep_msevec = results["P2_MAX"*string(nsamples)*"_FINAL"]["onestep_msevec"]
    filter!(x->x!==nothing, onestep_msevec)
    min_mse = minimum(onestep_msevec)
    push!(error, min_mse)
end
create_osa_plot(vnsamples, error, "2 link pendulum", "p2", "maximal")

error = Vector{Float64}()
for nsamples in vnsamples
    onestep_msevec = results["CP_MIN"*string(nsamples)*"_FINAL"]["onestep_msevec"]
    filter!(x->x!==nothing, onestep_msevec)
    min_mse = minimum(onestep_msevec)
    push!(error, min_mse)
end
create_osa_plot(vnsamples, error, "Cartpole", "cp", "minimal")

error = Vector{Float64}()
for nsamples in vnsamples
    onestep_msevec = results["P1_MIN"*string(nsamples)*"_FINAL"]["onestep_msevec"]
    filter!(x->x!==nothing, onestep_msevec)
    min_mse = minimum(onestep_msevec)
    push!(error, min_mse)
end
create_osa_plot(vnsamples, error, "1 link pendulum", "p1", "minimal")

error = Vector{Float64}()
for nsamples in vnsamples
    onestep_msevec = results["P2_MIN"*string(nsamples)*"_FINAL"]["onestep_msevec"]
    filter!(x->x!==nothing, onestep_msevec)
    min_mse = minimum(onestep_msevec)
    push!(error, min_mse)
end
create_osa_plot(vnsamples, error, "2 link pendulum", "p2", "minimal")