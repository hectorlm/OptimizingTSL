include("../../src/optimization/TRCG_NLS.jl")
include("../../src/config_utils.jl")
include("../../src/validation_metrics.jl")
using .ValidationMetrics
using Base.Threads
using Statistics
using Base: @kwdef
using JLD2

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
    return get_filename_configured(param, TCRLB, N_TSL, SNR, crit, mod)
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
mcrlb = ["CRLB" "MCRLB"];
crit = ["mean" "max"];
time_rate = ["T1rho" "R1rho"];

param_args = Vector{tsl_params}();
#for s in SNRs
#    for n in N_TSLs
#        for m in mcrlb # CRLB or MCRLB
#            for c in crit # Mean or Max Error
#                for r in time_rate # T1rho or R1rho
#                    # for mod in ["monoexp" "biexp" "stexp"]
#                    for mod in ["biexp"]
#                        st = get_filename(r,m,n,s,c,mod);
#                        file = matopen(st);
#                        temp = read(file, "spTSL_best")["TSL"];
#                        push!(param_args, tsl_params(mod,n,m,c,r,s,temp))
#                        close(file);
#                    end
#                end
#            end
#        end
#    end
#end

LogRange(x1,x2,n) = (10^y for y in range(log10(x1), log10(x2), length=n))
for s in SNRs
    for n in N_TSLs
        tsls = round.(collect(LinRange(1, 55.,n)))'
        tsls[2:end] = round.(tsls[2:end]);
        push!(param_args, tsl_params("biexp", n, "Non-Optimized", "Linear Spaced", "", s, tsls));
    end
end
for s in SNRs
    for n in N_TSLs
        tsls = round.(collect(LogRange(1, 55.,n)))'
        tsls[2:end] = round.(tsls[2:end]);
        push!(param_args, tsl_params("biexp", n, "Non-Optimized", "Log Spaced", "", s, tsls));
    end
end


trials = 1;
N = 100000;
forward = (p::Matrix{ComplexF64}, TSLs::Vector{Float64}) -> p[1, :].*(p[2, :].*exp.(-(TSLs'./p[3, :])) + (1 .-p[2, :]).*exp.(-(TSLs'./p[4, :])));


tab = Dict();
bds = [0.0 0.05 15 0.5; 1e10 0.95 100 10];
w = [0; 1; 1; 1];
tnrmse = zeros(length(param_args));
tmnae = zeros(length(param_args));

A = randn(ComplexF64, N);
A ./= abs.(A);
T1ρ = rand(Set(15:1:100), N);
T1ρ_s = rand(Set(0.5:0.1:10), N);
frac = rand(Set(0.05:0.01:0.95), N);
p = permutedims([A frac T1ρ T1ρ_s]);
ph = zeros(ComplexF64, (length(param_args), 4, N));

for inst in 1:length(param_args)
    p̂ = zeros(ComplexF64, (4,N));
    @info "Instance $inst"
    display(param_args[inst])
    print("\n")
    tsl = param_args[inst].TSLs[:];
    snr = param_args[inst].SNR;
    mvar = "T1rho";


    x = forward(p, tsl);
    y = x + (1/snr)*randn(ComplexF64, size(x));
    G = abs.(y[:, 1]);
    y ./= G;

    @time @threads for i in 1:N
        p̂[:, i] = nls_trcg(y[i, :], tsl, [1.0+0im; 0.5; 55; 5], bds, "biexp", mvar)[1];
    end
    
    p̂[1, :] .*= G;
    ph[inst, :, :] = p̂;
    tnrmse[inst] = nrmse(p̂, p, w);
    tmnae[inst] = nmae(p̂, p, w);
    println()
    @info "MNAE = $(tmnae[inst])"
    println(repeat("=",182))
end

tab = Dict("ModelParams" => param_args, "NRMSE"=>tnrmse,
             "MNAE"=>tmnae, "p" => p, "ph" => ph);
save("biexp_models_nonopt.jld2", tab)

