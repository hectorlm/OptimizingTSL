include("../src/optimization/TRCG_NLS.jl")
include("../src/config_utils.jl")
include("../src/validation_metrics.jl")
using .ValidationMetrics
using Base.Threads
using Plots
Plots.pyplot()
using Plots.PlotMeasures
using Statistics
using MAT
using Base: @kwdef
using JLD2

function get_filename(mod, crit)
    return "../../CAI2R_T1rho_phantom\\data\\Syn_noise\\images\\rec_images_$(mod)_$(crit)1.mat"
end

# Check if command line arguments are provided
if length(ARGS) == 0
    println("No command line arguments provided. Using default model = 'monoexp'")
    println("Usage: julia validation_phantom.jl <model>")
    println("Available models: monoexp, stexp, biexp")
    model = "monoexp"
else
    model = ARGS[1];
    println("Model: $model")
end

println(model)
# model = "biexp"; #monoexp, stexp or biexp
crits = ["lin", "log", "crlb", "mcrlb"];
crit_titles =["Linear Spaced" "Log Spaced" "CRLB" "MCRLB" "Ground Truth"];

L = 256;
if model == "monoexp"
    P = 1;
    parameters = ["Tmap"];
    parameter_titles = ["T1ρ        "];
    bds = [0.0 20; 1e10 70];
    w = [0; 1];
    sp = [1.0+0im; 45];
    lim = [(0, 70)];
elseif model == "stexp"
    P = 2;
    parameters = ["Tmap", "Bmap"];
    parameter_titles = ["T1ρ*       ", "β       "];
    bds = [0.0 20 0.1; 1e10 70 1.0];
    w = [0; 1; 1];
    sp = [1.0+0im; 45; 0.5];
    lim = [(0, 70) (0.0, 1.0)];
elseif model == "biexp"
    P = 3;
    parameters = ["Fmap", "Tmap", "Tsmap"]
    parameter_titles = ["f      ", "T1ρₗ        ", "T1ρₛ        "];
    bds = [0.0 0.05 15 0.1; 1e10 0.95 100 10.0];
    w = [0; 1; 1; 1];
    sp = [1.0+0im; 0.5; 55; 5.5];
    lim = [(0.0, 1.0) (0, 100) (0, 10)];
else
    error("Model not recognized")
end
C = length(crits)+1;

Map = zeros(ComplexF64, (L,L,P*C));
errors = zeros(Float64, (P*C));
for (i, crit) in enumerate(crits)
    st = get_filename(model, crit);
    file = matopen(st);
    data = read(file);
    close(file);

    tsl = data["TSL"][:];
    N_TSL = length(tsl);
    fimgs = reshape(data["recim"], (:,N_TSL));
    fmask = data["mask"][:];
    N = size(fimgs, 1);

    tnrmse = zeros(N);
    tmnae = zeros(N);
    p̂ = zeros(ComplexF64, (P+1,N));

    G = abs.(fimgs[:, 1]);
    # fimgs ./= G;
    @time @threads for i in findall(>(0),fmask)
        p̂[:, i] = nls_trcg(fimgs[i, :], tsl, sp, bds, model, "T1rho")[1];
    end

    mask = data["mask"];
    for j in 1:P
        if i == 1
            Map[:, :, j*C] = data[parameters[j]];
        end
        Map[:, :, i+(j-1)*C] = reshape(abs.(p̂[j+1,:]), size(mask));
        errors[i+(j-1)*C] = mnae(Map[mask, i+(j-1)*C], data[parameters[j]][mask])
    end
end 


cbs = permutedims([mod(i,C)==0 for i in 1:P*C]);
titles = [i <= C ? crit_titles[i] : "" for i in 1:P*C];
ylabels = [mod(i-1,C) == 0 ? parameter_titles[1+div(i,C)] : "" for i in 1:P*C];
xlabels = [mod(i, C)==0 ? "" : "MNAE=$(round.(errors[i], digits=4))" for i in 1:P*C];
lims = [lim[div(i-1,C)+1] for i in 1:P*C];

img = plot([heatmap(abs.(Map[:,:,i]), color=:jet, title=titles[i], ylabel=ylabels[i], yguidefontrotation = -90, xlabel=xlabels[i], clims=lims[i]) for i in 1:P*C]...,
    layout=(P,C), yflip=true, ticks=nothing, 
    colorbar=cbs, margin=10px, size=[1600,260*P+100])
savefig(img, "images//$(model).png")