"""
Validation Metrics Module

This module provides common validation metrics used across the TSL optimization project.
All metrics functions are defined here to avoid code duplication across validation scripts.
"""

module ValidationMetrics

using Statistics
using Random

export nmae, nrmse, cv, mnae

"""
    nmae(pred, truevalue, weights=1)

Calculate the Normalized Mean Absolute Error (NMAE).

# Arguments
- `pred`: Predicted values
- `truevalue`: True/reference values  
- `weights`: Optional weights (default=1 for unweighted)

# Returns
- NMAE value as a scalar

# Example
```julia
error = nmae(predictions, ground_truth)
weighted_error = nmae(predictions, ground_truth, weights)
```
"""
function nmae(pred, truevalue, weights=1)
    error = abs.(truevalue-pred)./abs.(truevalue);
    if weights == 1
        error = mean(error)
    else
        error = sum(weights.*mean(error, dims=2))/sum(weights)
    end
    return error
end

"""
    nmae(pred, truevalue)

Calculate the Normalized Mean Absolute Error (NMAE) - simplified version.

# Arguments
- `pred`: Predicted values
- `truevalue`: True/reference values

# Returns
- NMAE value as a scalar
"""
function nmae(pred, truevalue)
    return mean(abs.(truevalue-pred)./abs.(truevalue))
end

"""
    nrmse(pred, truevalue, weights=1)

Calculate the Normalized Root Mean Square Error (NRMSE).

# Arguments
- `pred`: Predicted values
- `truevalue`: True/reference values
- `weights`: Optional weights (default=1 for unweighted)

# Returns
- NRMSE value as a scalar

# Example
```julia
error = nrmse(predictions, ground_truth)
weighted_error = nrmse(predictions, ground_truth, weights)
```
"""
function nrmse(pred, truevalue, weights=1)
    error = mean(abs.(truevalue-pred).^2,dims=2);
    if weights == 1
        error = mean(sqrt.(error./mean(truevalue.^2, dims=2)))
    else
        error = sum(weights.*sqrt.(error./mean(abs.(truevalue).^2,dims=2)))/sum(weights)
    end
    return error
end

"""
    nrmse(pred, truevalue)

Calculate the Normalized Root Mean Square Error (NRMSE) - simplified version.

# Arguments
- `pred`: Predicted values
- `truevalue`: True/reference values

# Returns
- NRMSE value as a scalar
"""
function nrmse(pred, truevalue)
    error = sqrt.(sum(abs.(truevalue-pred).^2)./sum(abs.(truevalue).^2))
    return error
end

"""
    cv(pred, truevalue)

Calculate the Coefficient of Variation of the absolute error.

# Arguments
- `pred`: Predicted values
- `truevalue`: True/reference values

# Returns
- CV value as a scalar

# Example
```julia
variation = cv(predictions, ground_truth)
```
"""
function cv(pred, truevalue)
    return std((abs.(truevalue-pred)))./mean((abs.(truevalue-pred)))
end

"""
    mnae(pred, truevalue, weights=1)

Calculate the Mean Normalized Absolute Error (MNAE) - alias for nmae.

# Arguments
- `pred`: Predicted values
- `truevalue`: True/reference values
- `weights`: Optional weights (default=1 for unweighted)

# Returns
- MNAE value as a scalar
"""
function mnae(pred, truevalue, weights=1)
    return nmae(pred, truevalue, weights)
end

"""
    mnae(pred, truevalue)

Calculate the Mean Normalized Absolute Error (MNAE) - simplified version.
"""
function mnae(pred, truevalue)
    return nmae(pred, truevalue)
end

end # module ValidationMetrics
