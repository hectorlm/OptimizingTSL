using Pkg
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


trials = 1;
N = 2;
fw = (p::Matrix{ComplexF64}, TSLs::Vector{Float64}) -> p[1, :].*exp.(-(TSLs'./p[2, :]))

bds = [0.0 20; 1e10 70];
w = [0; 1];
tnrmse = zeros(length(param_args));
tmnae = zeros(length(param_args));
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

