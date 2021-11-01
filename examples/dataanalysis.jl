using Plots

include("utils.jl")


EXPERIMENT_IDs = ["P1_2D_MIN_GGK", "P1_2D_MAX_GGK", "P1_2D_PREV_VE", "P2_2D_MIN_GGK", "P2_2D_MAX_GGK"]

for EXPERIMENT_ID in EXPERIMENT_IDs
    success, checkpointdict = loadcheckpoint(EXPERIMENT_ID*"_FINAL")
    !success && println("No checkpoint found. Please check the experiment ID")

    onestep_msevec = checkpointdict["onestep_msevec"]
    onestep_params = checkpointdict["onestep_params"]
    onestep_params = onestep_params[onestep_msevec .!== nothing]
    onestep_msevec = onestep_msevec[onestep_msevec .!== nothing]
    println("$EXPERIMENT_ID : Best onestep MSE: $(minimum(onestep_msevec))")
end

#=
data = [1e-3, 3e-4, 2e-4, 2e-4]
labels = ["0 v prediction" "last v prediction" "GP min" "GP max"]
osa_bar_plot = bar(data,
                   xticks=(1:4, labels),
                   ylabel="One Step Ahead forecast error",
                   label="",
                   title="OSA error comparison !NOT ACTUAL DATA!")
png(osa_bar_plot, "barplot_OSAerror")

steps = 100
cumdata = []
for id in 1:length(data)
    push!(cumdata, data[id] .+ randn(steps).*0.01*data[id])
end

osa_series_plot = plot(1:steps, cumdata,
                       xlabel="Steps",
                       ylabel="One Step Ahead forecast error",
                       label=labels,
                       leg=:left,
                       title="OSA error comparison !NOT ACTUAL DATA!")
png(osa_series_plot, "seriesplot_OSAerror")

data = [1e-1, 3e-2, 5e-3, 5e-3]
labels = ["0 v prediction" "last v prediction" "GP min" "GP max"]
osa_bar_plot = bar(data,
                   xticks=(1:4, labels),
                   ylabel="Simulation error",
                   label="",
                   title="Simulation error comparison !NOT ACTUAL DATA!")
png(osa_bar_plot, "barplot_SIMerror")

steps = 100
cumdata = []
for id in 1:length(data)
    push!(cumdata, cumsum(data[id]/steps .+ randn(steps).*0.01*data[id]))
end

osa_series_plot = plot(1:steps, cumdata,
                       xlabel="Steps",
                       ylabel="One Step Ahead forecast error",
                       label=labels,
                       leg=:left,
                       title="Simulation error comparison !NOT ACTUAL DATA!")
png(osa_series_plot, "seriesplot_SIMerror")
=#