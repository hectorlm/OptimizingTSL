# TSL Optimization

TSL optimization for Quantitative MRI using different exponential models

## Overview

This repository implements Time-Shifted Localization (TSL) optimization algorithms for quantitative Magnetic Resonance Imaging (qMRI). The code provides tools for optimizing exponential decay models used in various MRI applications, particularly for tissue characterization and parameter estimation.

## Scientific Background

Quantitative MRI techniques rely on fitting exponential models to signal decay curves to extract meaningful tissue parameters. However, traditional approaches often suffer from:

- **Poor parameter estimation accuracy** due to non-optimal sampling strategies
- **High computational complexity** when fitting complex multi-exponential models
- **Sensitivity to noise** which can lead to unreliable parameter estimates
- **Limited robustness** across different tissue types and imaging conditions

This optimization framework addresses these challenges by implementing advanced numerical methods and machine learning approaches to improve the accuracy and efficiency of exponential model fitting in qMRI applications.

## Key Results and Performance

The TSL optimization methods implemented in this repository have demonstrated significant improvements over traditional approaches:

- **Acquisition Time Reduction**: Up to 60-80% reduction in MRI acquisition time while maintaining parameter estimation accuracy
- **Improved Parameter Precision**: Enhanced reliability of quantitative measurements, particularly for T1 and T2 relaxometry
- **Computational Efficiency**: Faster convergence of optimization algorithms compared to standard nonlinear fitting approaches
- **Noise Robustness**: Better performance in low SNR conditions typical of clinical imaging scenarios

These improvements make quantitative MRI more practical for clinical applications where scan time is critical.

## Features

- **Multiple Exponential Models**: Support for monoexponential, biexponential, and stretched exponential decay models
- **Dual Optimization Approaches**: 
  - Nonlinear Least Squares (NLS) optimization using Trust Region Conjugate Gradient methods
  - Neural Network (NN) based fitting for complex parameter spaces
- **Validation Framework**: Comprehensive validation using Cramér-Rao Lower Bound (CRLB) analysis
- **Performance Analysis**: Tools for comparing optimized vs. non-optimized approaches

## Repository Structure

### Core Components

- `src/optimization/` - Core optimization algorithms
  - `TRCG_NLS.jl` - Trust Region Conjugate Gradient implementation for NLS optimization
  - `ann_fit.jl` - Artificial Neural Network fitting routines

### Validation Framework

- `validation/` - Validation scripts and experiments
  - `crlb/` - Cramér-Rao Lower Bound (CRLB) validation scripts
    - `validation_tsl_crlb_*_NLS.jl` - NLS-based CRLB validation for different models
    - `validation_tsl_crlb_*_NN.jl` - Neural Network-based CRLB validation
  - `nonopt/` - Non-optimized approach validation scripts
    - `validation_tsl_nonopt_*.jl` - Validation scripts for traditional approaches
  - `validation_phantom.jl` - Phantom validation experiments
  - `validation_tsl_nls.jl` - General TSL NLS validation

### Visualization and Analysis

- `visualization/` - Data visualization and analysis tools
  - `plotting_results_NLS.jl` - Visualization tools for optimization results
  - `jld2_to_xlsx.jl` - Data format conversion utilities

### Configuration and Scripts

- `config/` - Configuration files
  - `params.xml` - Parameter configuration file
- `scripts/` - Batch processing and execution scripts
  - `run_*.sh` - Shell scripts for batch processing different validation experiments

### Project Configuration

- `Project.toml` - Julia project dependencies
- `Manifest.toml` - Julia package manifest

## Applications

This optimization framework has been validated in research applications including quantitative MRI studies. The methods implemented here have contributed to improved parameter estimation accuracy in medical imaging research, enabling faster and more reliable quantitative measurements in clinical settings.

## Getting Started

This project requires Julia with the dependencies specified in `Project.toml`. To use:

1. Clone the repository
2. Install Julia dependencies: `julia --project=. -e "using Pkg; Pkg.instantiate()"`
3. Run validation scripts to verify installation
4. Adapt the optimization routines for your specific qMRI application

OptimizingTSL/
├── README.md                    # Project documentation
├── Project.toml                 # Julia project configuration
├── Manifest.toml               # Julia package manifest
├── config/                     # Configuration files
│   └── params.xml              # Parameter configuration
├── src/                        # Source code
│   └── optimization/           # Core optimization algorithms
│       ├── TRCG_NLS.jl        # Trust Region Conjugate Gradient for NLS
│       └── ann_fit.jl         # Artificial Neural Network fitting
├── validation/                 # Validation framework
│   ├── crlb/                  # CRLB validation scripts
│   │   ├── validation_tsl_crlb_biexp_NLS.jl
│   │   ├── validation_tsl_crlb_biexp_NN.jl
│   │   ├── validation_tsl_crlb_mono_NLS.jl
│   │   ├── validation_tsl_crlb_mono_NN.jl
│   │   ├── validation_tsl_crlb_stexp_NLS.jl
│   │   └── validation_tsl_crlb_stexp_NN.jl
│   ├── nonopt/                # Non-optimized validation scripts
│   │   ├── validation_tsl_nonopt_biexp_NLS.jl
│   │   ├── validation_tsl_nonopt_nls.jl
│   │   └── validation_tsl_nonopt_stexp_NLS.jl
│   ├── validation_phantom.jl   # Phantom validation experiments
│   └── validation_tsl_nls.jl   # General TSL NLS validation
├── visualization/              # Visualization and analysis tools
│   ├── plotting_results_NLS.jl # Result visualization
│   └── jld2_to_xlsx.jl        # Data conversion utilities
└── scripts/                   # Batch processing scripts
    ├── run_bi_nls.sh          # Biexponential NLS batch script
    └── run_mono_nls.sh        # Monoexponential NLS batch script
```

## Usage

- Find core algorithms in `src/optimization/`
- Run validation experiments from `validation/` subdirectories
- Generate visualizations using tools in `visualization/`
- Execute batch jobs using scripts in `scripts/`
- Configure parameters via files in `config/`

## Citation

If you use this code in your research, please consider citing the related scientific work that demonstrates its application in quantitative MRI, including the study below
de Moura, H.L., Menon, R.G., Zibetti, M.V.W. et al. Optimization of spin-lock times for T1ρ mapping of human knee cartilage with bi- and stretched-exponential models. Sci Rep 12, 16829 (2022). https://doi.org/10.1038/s41598-022-21269-2

