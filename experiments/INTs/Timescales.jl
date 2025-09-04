# using Base.Threads

# # Check thread configuration
# println("System CPU threads: ", Sys.CPU_THREADS)
# println("Julia threads: ", nthreads())

# # If you want to ensure you're using all available cores
# if nthreads() == 1
#     println("Warning: Julia is running with only 1 thread!")
#     println("Restart Julia with: julia --threads=auto")
# end

using IntrinsicTimescales
using NPZ
using Statistics
using Base.Threads
using HDF5  # Add this

# Load the simulation data
data_homogeneous_path = "/home/frank/HBNM/experiments/INTs/INT_Outputs/homogeneous_simulations.npz"
data_heterogeneous_path = "/home/frank/HBNM/experiments/INTs/INT_Outputs/heterogeneous_simulations.npz"

homogeneous = npzread(data_homogeneous_path)
heterogeneous = npzread(data_heterogeneous_path)
data_homogeneous = homogeneous["data"]
data_heterogeneous = heterogeneous["data"]

println("Loaded data with shape: ", size(data_homogeneous))

# Set the sampling rate
const fs = 1000.0

# Get dimensions
n_particles, n_iterations, n_regions, n_timepoints = size(data_homogeneous)

# Preallocate for fixed-size results
acw50_values_heterogeneous = zeros(Float64, n_particles, n_regions)
tau_values_heterogeneous = zeros(Float64, n_particles, n_regions)
auc_values_heterogeneous = zeros(Float64, n_particles, n_regions)

# Use Vector for variable-size results (ACF/PSD)
acf_values_homogeneous = Vector{Vector{Float64}}(undef, n_particles)
psd_values_homogeneous = Vector{Vector{Float64}}(undef, n_particles)
acf_values_heterogeneous = Vector{Vector{Float64}}(undef, n_particles)
psd_values_heterogeneous = Vector{Vector{Float64}}(undef, n_particles)

# Process homogeneous model
results = acw(data_homogeneous[1, :, :, :], fs, 
acwtypes=[:acw50, :tau, :auc], 
dims=3,           # time is the 3rd dimension
trial_dims=1,     # iterations are the 1st dimension  
parallel=true, 
average_over_trials=true,
return_acf=true, 
return_psd=true)

# Plot the ACF (Shape 1, 180, 17)
using Plots
# Plot all 180 regions' ACFs on the same figure, each with a different color
lags = 1:size(results.acf, 3)
plot()
for region in 1:180
    plot!(lags, results.acf[1, region, :], label=false)
end
xlabel!("Lag")
ylabel!("Correlation")
savefig("acf_plot.png")


# Plot using convenience function
acwplot(results, only_acf=true)


acw50_values_homogeneous[particle, :] = results.acw_results[1]
tau_values_homogeneous[particle, :] = results.acw_results[2]
auc_values_homogeneous[particle, :] = results.acw_results[3]
acf_values_homogeneous[particle] = results.acf
psd_values_homogeneous[particle] = results.psd


# Save results for Homogeneous Model using HDF5
h5open("/home/frank/HBNM/experiments/INTs/INT_Outputs/timescales_results_homogeneous.h5", "w") do file
    file["acw50_homogeneous"] = acw50_values_homogeneous
    file["tau_homogeneous"] = tau_values_homogeneous
    file["auc_homogeneous"] = auc_values_homogeneous
    file["acf_homogeneous"] = acf_values_homogeneous
    file["psd_homogeneous"] = psd_values_homogeneous
end

println("Homogeneous Model results saved successfully!")

# Process heterogeneous model
println("Computing timescales for heterogeneous model...")
@threads for particle in 1:n_particles
    results = acw(data_heterogeneous[particle, :, :, :], fs, 
                  acwtypes=[:acw50, :tau, :auc], 
                  dims=3, 
                  trial_dims=1,
                  return_acf=true, 
                  return_psd=true,
                  parallel=true, 
                  average_over_trials=true)
    
    acw50_values_heterogeneous[particle, :] = results.acw_results[1]
    tau_values_heterogeneous[particle, :] = results.acw_results[2]
    auc_values_heterogeneous[particle, :] = results.acw_results[3]
    acf_values_heterogeneous[particle] = results.acf
    psd_values_heterogeneous[particle] = results.psd
end

# Save results for Heterogeneous Model using HDF5
h5open("/home/frank/HBNM/experiments/INTs/INT_Outputs/timescales_results_heterogeneous.h5", "w") do file
    file["acw50_heterogeneous"] = acw50_values_heterogeneous
    file["tau_heterogeneous"] = tau_values_heterogeneous
    file["auc_heterogeneous"] = auc_values_heterogeneous
    file["acf_heterogeneous"] = acf_values_heterogeneous
    file["psd_heterogeneous"] = psd_values_heterogeneous
end

println("Heterogeneous Model results saved successfully!")