include("../src/optimization/TRCG_NLS.jl")
include("../src/config_utils.jl")
include("../src/validation_metrics.jl")
using .ValidationMetrics
using Base.Threads
using Statistics
using MAT
using Base: @kwdef
using JLD2
using Random

function gen_monoexp_data(N)
    Random.seed!(30);
    A = randn(ComplexF64, N);
    A ./= abs.(A);
    T1ρ = collect(20:1:70.0);
    T1ρ_sampled = rand(T1ρ, N);
    return [A'; T1ρ_sampled']  # 2×N matrix
end
function gen_stexp_data(N)
    T1ρ = 20:1:80.0;
    Betas = 0.4:0.1:1.0;
    T1ρ_g = [x for x in T1ρ, y in Betas];
    Betas_g = [y for x in T1ρ, y in Betas];
    A = randn(ComplexF64, length(T1ρ_g));
    A ./= abs.(A);
    return repeat([permutedims(A); T1ρ_g[:]'; Betas_g[:]'], 1, N);
end

function gen_biexp_data(N)
    frac = 0.05:0.1:0.95;
    t1rl = 30:1:80.0;
    t1rs = 1:1:20.0;
    frac_g = [x for x in frac, y in t1rl, z in t1rs];
    t1rl_g = [y for x in frac, y in t1rl, z in t1rs];
    t1rs_g = [z for x in frac, y in t1rl, z in t1rs];
    A = randn(ComplexF64, length(frac_g));
    A ./= abs.(A);
    return repeat([permutedims(A); frac_g[:]'; t1rl_g[:]'; t1rs_g[:]'],1,N);
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

function get_filename(param, TCRLB, N_TSL, SNR, crit, model)
    return get_filename_configured_eggs(param, TCRLB, N_TSL, SNR, crit, model)
end

mods = [];
snrs = [];
crits = [];
optvars = [];
time_rates = [];
for mod in ["monoexp" "stexp" "biexp" "comb"]
    for s in [30 125]
        for m in ["CRLB", "MCRLB"] # CRLB or MCRLB
            for c in ["mean", "max"] # Mean or Max Error
                for r in ["T1rho", "R1rho"] # T1rho or R1rho
                    push!(mods, mod);
                    push!(snrs, s);
                    push!(crits, m);
                    push!(optvars,c);
                    push!(time_rates, r);
                end
            end
        end
    end
end

i = parse(Int64, ARGS[1]);
println("Job id $i")
model = mods[i];
m = crits[i];
s = snrs[i];
c = optvars[i];
r = time_rates[i];
param_args = Vector{tsl_params}();

if model == "monoexp"
    P = 2;
    parameters = ["Tmap"];
    parameter_titles = ["T1ρ        "];
    bds = [0.0 20; 1e10 70];
    w = [0; 1]; 
    sp = [1.0+0im; 45];
    lim = [(0, 70)];
    fw = (p::Matrix{ComplexF64}, TSLs::Vector{Float64}) -> p[1, :].*exp.(-(TSLs'./p[2, :]))
    N = 1000;
    p = gen_monoexp_data(N);
    N_TSLs = [2 3 4 5 6 7 8];
elseif model == "stexp"
    P = 3;
    parameters = ["Tmap", "Bmap"];
    parameter_titles = ["T1ρ*       ", "β       "];
    bds = [0.0 10 0.4; 1e10 90 1.0];
    w = [0; .9; .1];
    sp = [1.0+0im; 55; 0.9];
    lim = [(0, 55) (0.0, 1.0)];
    fw = (p::Matrix{ComplexF64}, TSLs::Vector{Float64}) -> p[1, :].*exp.(-(TSLs'./p[2, :]).^p[3, :]);
    N = 1000;
    p = gen_stexp_data(N);
    N_TSLs = [3 4 5 6 7 8 9];
elseif model == "biexp"
    P = 4;
    parameters = ["Fmap", "Tmap", "Tsmap"]
    parameter_titles = ["f      ", "T1ρₗ        ", "T1ρₛ        "];
    bds = [0.0 0.05 30 1; 1e10 0.95 100 20.0];
    w = [0.; 0.4; 0.3; 0.3];
    sp = [1.0+0im; 0.5; 55; 5];
    lim = [(0.0, 1.0) (30, 100) (0, 20)];
    fw = (p::Matrix{ComplexF64}, TSLs::Vector{Float64}) -> p[1, :].*(p[2, :].*exp.(-(TSLs'./p[3, :])) + (1 .-p[2, :]).*exp.(-(TSLs'./p[4, :])));
    N = 700;
    p = gen_biexp_data(N);
    N_TSLs = [4 5 6 7 8 9 10 11 12];
else
    error("Model not recognized")
end
w ./= sum(w);


for n in N_TSLs
    st = get_filename(r,m,n,s,c,model);
    file = matopen(st);
    temp = read(file, "spTSL_best")["TSL"];
    push!(param_args, tsl_params(model,n,m,c,r,s,temp))
    close(file);
end

tnrmse = zeros((P, length(param_args)));
tmnae = zeros((P, length(param_args)));
tcv = zeros((P, length(param_args)));
tstd = zeros((P, length(param_args)));
NL = size(p,2);
for inst in 1:length(param_args)
    p̂ = zeros(ComplexF64, (P,NL));
    println("Instance $inst")
    display(param_args[inst])
    print("\n")
    tsl = param_args[inst].TSLs[:];
    snr = param_args[inst].SNR;
    println("SNR = $(snr)dB")
    mvar = param_args[inst].model_var;

    x = fw(p, tsl);
    y = x + (1/snr)*randn(ComplexF64, size(x));
    G = abs.(y[:, 1]);
    y ./= G;

    @time @threads for i in 1:NL
        p̂[:, i] = nls_trcg(y[i, :], tsl, sp, bds, model, "T1rho"; maxiter=10000)[1];
    end
    
    p̂[1, :] .*= G;
    for pid in 1:P
        tnrmse[pid, inst] = nrmse(p̂[pid, :], p[pid, :]);
        tmnae[pid, inst] = nmae(p̂[pid, :], p[pid, :]);
        tcv[pid, inst] = cv(p̂[pid, :], p[pid, :]);
    end
    tstd[:, inst] = std(abs.(p̂ - p)./abs.(p),dims=2);
    println("Bias")
    println(mean(abs.(p̂.-p),dims=2))
    println("Std. Dev.")
    println(std(abs.(p̂.-p)./abs.(p),dims=2))

    println()
    println( "NRMSE = $(sum(w.*tnrmse[:,inst]))")
    println("MNAE = $(sum(w.*tmnae[:, inst]))")
    println(repeat("=",182))
end

tab = Dict("ModelParams" => param_args, "NRMSE"=>tnrmse,
             "MNAE"=>tmnae, "CV"=>tcv, "STD"=>tstd);
save("data/$(model)_$(s)_$(m)_$(c)_$(r).jld2", tab)

# end
