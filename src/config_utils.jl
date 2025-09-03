"""
Configuration utilities for TSL Optimization project.
Provides centralized access to configuration parameters including data paths.
"""

# Install LightXML if not already installed
# Pkg.add("LightXML")
using LightXML

"""
    get_config_param(param_name::String, config_file::String = nothing)

Read a configuration parameter from the params.xml file.

# Arguments
- `param_name::String`: Name of the parameter to retrieve
- `config_file::String`: Path to the configuration file (auto-determined if nothing)

# Returns
- The parameter value as a string, or nothing if not found

# Example
```julia
data_path = get_config_param("data_path")
```
"""
function get_config_param(param_name::String, config_file = nothing)
    # Auto-determine config file path based on call location
    if config_file === nothing
        # Try different relative paths based on common calling locations
        potential_paths = [
            "../../config/params.xml",  # From validation/crlb/ or validation/nonopt/
            "../config/params.xml",     # From validation/
            "config/params.xml"         # From root
        ]
        
        config_file = nothing
        for path in potential_paths
            if isfile(path)
                config_file = path
                break
            end
        end
        
        if config_file === nothing
            @error "Could not find params.xml configuration file"
            return nothing
        end
    end
    
    try
        xdoc = parse_file(config_file)
        xroot = root(xdoc)
        
        # Find the parameter
        param_nodes = get_elements_by_tagname(xroot, param_name)
        if !isempty(param_nodes)
            result = content(param_nodes[1])
            free(xdoc)
            return result
        else
            free(xdoc)
            @warn "Parameter '$param_name' not found in configuration file"
            return nothing
        end
    catch e
        @error "Error reading configuration file '$config_file': $e"
        return nothing
    end
end

"""
    get_data_path(config_file::String = nothing)

Get the configured data path for TSL sampling times.

# Arguments
- `config_file::String`: Path to the configuration file (auto-determined if nothing)

# Returns
- The data path as a string

# Example
```julia
data_path = get_data_path()
```
"""
function get_data_path(config_file = nothing)
    path = get_config_param("data_path", config_file)
    if path === nothing
        @warn "Data path not configured, using default"
        return "../TSL_opt_Hector/sampling_times/"
    end
    return path
end

"""
    get_filename_configured(param, TCRLB, N_TSL, SNR, crit, mod, config_file::String = nothing)

Generate a filename using the configured data path.

# Arguments
- `param`: Parameter type (e.g., "T1rho", "R1rho")
- `TCRLB`: CRLB type (e.g., "CRLB", "MCRLB")
- `N_TSL`: Number of TSL points
- `SNR`: Signal-to-noise ratio
- `crit`: Optimization criterion (e.g., "mean", "max")
- `mod`: Model type (e.g., "monoexp", "biexp", "stexp")
- `config_file::String`: Path to the configuration file (auto-determined if nothing)

# Returns
- Complete file path as a string

# Example
```julia
filename = get_filename_configured("T1rho", "CRLB", 5, 30, "mean", "biexp")
```
"""
function get_filename_configured(param, TCRLB, N_TSL, SNR, crit, mod, config_file = nothing)
    data_path = get_data_path(config_file)
    return "$(data_path)TSLs_$(param)_$(TCRLB)_N$(N_TSL)_SNR$(SNR)_$(crit)_$(mod).mat"
end

"""
    get_filename_configured_eggs(param, TCRLB, N_TSL, SNR, crit, mod, config_file::String = nothing)

Generate a filename with "eggs" prefix using the configured data path.

# Arguments
- Same as get_filename_configured

# Returns
- Complete file path as a string with "eggs" prefix

# Example
```julia
filename = get_filename_configured_eggs("T1rho", "CRLB", 5, 30, "mean", "biexp")
```
"""
function get_filename_configured_eggs(param, TCRLB, N_TSL, SNR, crit, mod, config_file = nothing)
    data_path = get_data_path(config_file)
    return "$(data_path)TSLs_eggs_$(param)_$(TCRLB)_N$(N_TSL)_SNR$(SNR)_$(crit)_$(mod).mat"
end

export get_config_param, get_data_path, get_filename_configured, get_filename_configured_eggs
