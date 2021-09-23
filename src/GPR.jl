module GPR

using StaticArrays
using LinearAlgebra
using Statistics
using Plots

export GaussianProcessRegressor
export MOGaussianProcessRegressor
export AbstractKernel
export GaussianKernel
export GeneralGaussianKernel
export MaternKernel
export predict
export predict_full
export plot_gp

include(joinpath("kernels", "AbstractKernel.jl"))
include(joinpath("utils", "cholesky.jl"))
include(joinpath("visualization", "visualization.jl"))
include(joinpath("kernels", "GaussianKernel.jl"))
include(joinpath("kernels", "MaternKernel.jl"))
include(joinpath("utils", "kernelmatrix.jl"))
include(joinpath("regressors", "gp_regressor.jl"))
include(joinpath("regressors", "mogp_regressor.jl"))

end