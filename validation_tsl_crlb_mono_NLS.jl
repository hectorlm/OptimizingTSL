using Pkg
# Pkg.add("Statistics")
# Pkg.add("MAT")
# Pkg.add("LinearAlgebra")
# Pkg.add("JLD2")
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

SNRs = [30];
N_TSLs = [4 5 6 7 8 9 10];
mcrlb = ["CRLB", "MCRLB"];
crit = ["mean"];
time_rate = ["T1rho"];

param_args = Vector{tsl_params}();
for s in SNRs
    for m in mcrlb # CRLB or MCRLB
        for n in N_TSLs
            for c in crit # Mean or Max Error
                for r in time_rate # T1rho or R1rho
                    # for mod in ["monoexp" "biexp" "stexp"]
                    for mod in ["monoexp"]
                        st = get_filename(r,m,n,s,c,mod);
                        file = matopen(st);
                        temp = read(file, "spTSL_best")["TSL"];
                        push!(param_args, tsl_params(mod,n,m,c,r,s,temp))
                        close(file);
                    end
                end
            end
        end
    end
end

# LogRange(x1,x2,n) = round.(10^y for y in range(log10(x1+1), log10(x2+1), length=n)).-1
# for s in SNRs
#     for n in N_TSLs
#         push!(param_args, tsl_params("monoexp", n, "Non-Optimized", "Linear Spaced", "", s, round.(collect(LinRange(0., 55.,n)'))));
#     end
# end
# for s in SNRs
#     for n in N_TSLs
#         push!(param_args, tsl_params("monoexp", n, "Non-Optimized", "Log Spaced", "", s, LogRange(0., 55. ,n)'));
#     end
# end

trials = 1;
N = 2;
fw = (p::Matrix{ComplexF64}, TSLs::Vector{Float64}) -> p[1, :].*exp.(-(TSLs'./p[2, :]))


# tab = Dict();
# R2i = zeros(length(param_args));
# model_loss = zeros(length(param_args));
# model_nrmse = zeros(length(param_args));
# model_nmae = zeros(length(param_args));

# try
#     global tab = load("st_models.jld2");
#     global R2i = tab["R2"];
#     global model_loss = tab["Loss"];
#     global model_nrmse = tab["NRMSE"];
#     global model_nmae = tab["MNAE"];
# catch ArgumentError
#     pass;
# end

bds = [0.0 20; 1e10 70];
w = [0; 1];
tnrmse = zeros(length(param_args));
tmnae = zeros(length(param_args));
# p = zeros(ComplexF64, (2,N));
# p̂ = zeros(ComplexF64, (2,N));
for inst in 1:length(param_args)
    # global p, p̂
    p = zeros(ComplexF64, (2,N));
    p̂ = zeros(ComplexF64, (2,N));
    @info "Instance $inst"
    display(param_args[inst])
    print("\n")
    tsl = max.(1, param_args[inst].TSLs[:]);
    snr = param_args[inst].SNR;
    mvar = param_args[inst].model_var;

    A = randn(ComplexF64, N);
    A ./= abs.(A);
    T1ρ = rand(Set(20:0.1:70), N);
    p = permutedims([A T1ρ])

    x = fw(p, tsl);
    y = x + (1/snr)*randn(ComplexF64, size(x));
    G = abs.(y[:, 1]);
    y ./= G;

    @time for i in 1:N
        p̂[:, i] = nls_trcg(y[i, :], tsl, [1.0+0im; 45], bds, "monoexp", mvar)[1];
    end
    
    p̂[1, :] .*= G;
    tnrmse[inst] = nrmse(p̂, p, w);
    tmnae[inst] = nmae(p̂, p, w);
    println()
    @info "NRMSE = $(tnrmse[inst])"
    @info "MNAE = $(tmnae[inst])"
    println(repeat("=",182))
end

tab = Dict("ModelParams" => param_args, "NRMSE"=>tnrmse,
             "MNAE"=>tmnae);
save("mono_models.jld2", tab)

