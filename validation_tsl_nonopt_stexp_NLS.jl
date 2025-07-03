#using Pkg
#Pkg.add("Statistics")
#Pkg.add("MAT")
#Pkg.add("LinearAlgebra")
#Pkg.add("JLD2")
#include("TRCG_NLS.jl")
include("TRCG_NLS.jl")
using Base.Threads
# using Plots
using Statistics
using MAT
using Base: @kwdef
# using LightXML
# using Printf: @sprintf
using JLD2
# plotlyjs()

function nmae(pred, truevalue, weights=1)
    error = abs.(truevalue-pred)./abs.(truevalue);
    if weights == 1
        error = mean(error)
    else
        error = sum(weights.*mean(error, dims=2))/sum(weights)
    end
    return error
end

function nrmse(pred, truevalue, weights=1)
    error = mean(abs.(truevalue-pred).^2,dims=2);
    if weights == 1
        error = mean(sqrt.(error./mean(truevalue.^2, dims=2)))
    else
        error = sum(weights.*sqrt.(error./mean(abs.(truevalue).^2,dims=2)))/sum(weights)
    end
    return error
end

@kwdef mutable struct tsl_params
    model::String
    N_TSLs::Int
    opt_type::String
    opt_crit::String
    model_var::String
    SNR::Int
    TSLs::Matrix{Float64}
end

function get_filename(param, TCRLB, N_TSL, SNR, crit, mod)
    return "..//TSL_opt_Hector//sampling_times//TSLs_$(param)_$(TCRLB)_N$(N_TSL)_SNR$(SNR)_$(crit)_$(mod).mat"
end

# function get_param_xml(filepath)
#     xdoc = parse_file(filepath);
#     xroot = root(xdoc);
#     SNR = parse.(Int, content.(xroot["SNR"]));
#     N_TSLs = parse.(Int, content.(xroot["TSLs"]));
#     mcrlb = content.(xroot["mcrlb"]);
#     crit = content.(xroot["crit"]);
#     time_rate = content.(xroot["time_rate"]);
#     free(xdoc)
#     return SNR, N_TSLs, mcrlb, crit, time_rate
# end

# SNRs, N_TSLs, mcrlb, crit, time_rate = get_param_xml(".//params.xml")

SNRs = [30, 125];
N_TSLs = [3 4 5 6 7 8 9 10];
mcrlb = ["CRLB", "MCRLB"];
crit = ["mean", "max"];
time_rate = ["T1rho", "R1rho"];

param_args = Vector{tsl_params}();

LogRange(x1,x2,n) = round.(10^y for y in range(log10(x1+1), log10(x2+1), length=n)).-1
for s in SNRs
    for n in N_TSLs
        push!(param_args, tsl_params("stexp", n, "Non-Optimized", "Linear Spaced", "", s, round.(collect(LinRange(1., 55.,n)'))));
    end
end
for s in SNRs
    for n in N_TSLs
        push!(param_args, tsl_params("stexp", n, "Non-Optimized", "Log Spaced", "", s, LogRange(1., 55. ,n)'));
    end
end

trials = 1;
N = 100000;
forward = (p::Matrix{ComplexF64}, TSLs::Vector{Float64}) -> p[1, :].*exp.(-(TSLs'./p[2, :]).^p[3, :]);

bds = [0.0 20 0.1; 1e10 70 1.0];
w = [0; 1; 1];
tnrmse = zeros(length(param_args));
tmnae = zeros(length(param_args));

A = randn(ComplexF64, N);
A ./= abs.(A);
T1ρ = rand(Set(20:0.1:70), N);
Betas = rand(Set(0.1:0.01:1.0), N);
p = permutedims([A T1ρ Betas]);
ph = zeros(ComplexF64, (length(param_args), 4, N));

for inst in 1:length(param_args)
    # p = zeros(ComplexF64, (3,N));
    p̂ = zeros(ComplexF64, (3,N));
    
    #global p, p̂
    #@info "Instance $inst"
    #display(param_args[inst])
    #print("\n")
    tsl = param_args[inst].TSLs[:];
    snr = param_args[inst].SNR;
    mvar = "T1rho";#param_args[inst].model_var;


    x = forward(p, tsl);
    y = x + (1/snr)*randn(ComplexF64, size(x));
    G = abs.(y[:, 1]);
    y ./= G;

    @threads for i in 1:N
        p̂[:, i] = nls_trcg(y[i, :], tsl, [1.0+0im; 45; 0.8], bds, "stexp", mvar)[1];
    end
    
    p̂[1, :] .*= G;
    ph[inst, :, :] = p̂;
    tnrmse[inst] = nrmse(p̂, p, w);
    tmnae[inst] = nmae(p̂, p, w);
    #println()
    #@info "NRMSE = $(tnrmse[inst])"
    #@info "MNAE = $(tmnae[inst])"
    #println(repeat("=",182))
end

# plot(N_TSLs[:], [tmnae, tnrmse])
tab = Dict("ModelParams" => param_args, "NRMSE"=>tnrmse,
"MNAE"=>tmnae, "p" => p, "ph" => ph);
save("stexp_models_nonopt.jld2", tab)

