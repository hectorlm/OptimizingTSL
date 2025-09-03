include("../../src/optimization/TRCG_NLS.jl")
include("../../src/config_utils.jl")
include("../../src/validation_metrics.jl")
using .ValidationMetrics
using Base.Threads
using Statistics
using MAT
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

SNRs = [30, 125];
N_TSLs = [3 4 5 6 7 8 9 10];
mcrlb = ["CRLB", "MCRLB"];
crit = ["mean", "max"];
time_rate = ["T1rho", "R1rho"];

param_args = Vector{tsl_params}();
for s in SNRs
    for n in N_TSLs
        for m in mcrlb # CRLB or MCRLB
            for c in crit # Mean or Max Error
                for r in time_rate # T1rho or R1rho
                    # for mod in ["monoexp" "biexp" "stexp"]
                    for mod in ["stexp"]
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
    p̂ = zeros(ComplexF64, (3,N));

    tsl = param_args[inst].TSLs[:];
    snr = param_args[inst].SNR;
    mvar = "T1rho";


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
end

tab = Dict("ModelParams" => param_args, "NRMSE"=>tnrmse,
"MNAE"=>tmnae, "p" => p, "ph" => ph);
save("stexp_models_$N.jld2", tab)

