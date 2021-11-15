include("TRCG_NLS.jl")
using Base.Threads
using Plots
Plots.pyplot()
using Plots.PlotMeasures
using Statistics
using MAT
using Base: @kwdef
using JLD2

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

function get_filename(mod, crit)
    return "../../CAI2R_T1rho_phantom\\data\\Syn_noise\\images\\rec_images_$(mod)_$(crit)1.mat"
end

model = "biexp"; #monoexp, stexp or biexp
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
elseif model == "stexp"
    P = 2;
    parameters = ["Tmap", "Bmap"];
    parameter_titles = ["T1ρ*       ", "β       "];
    bds = [0.0 20 0.1; 1e10 70 1.0];
    w = [0; 1; 1];
    sp = [1.0+0im; 45; 0.5];
elseif model == "biexp"
    P = 3;
    parameters = ["Fmap", "Tmap", "Tsmap"]
    parameter_titles = ["f      ", "T1ρₗ        ", "T1ρₛ        "];
    bds = [0.0 0.05 15 0.1; 1e10 0.95 100 10.0];
    w = [0; 1; 1; 1];
    sp = [1.0+0im; 0.5; 55; 5.5];
else
    error("Model not recognized")
end
C = length(crits)+1;

Map = zeros(ComplexF64, (L,L,P*C))

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
        # Map[:, :, i] = reshape(abs.(p̂[j,:]), size(mask));
        Map[:, :, i+(j-1)*C] = reshape(abs.(p̂[j+1,:]), size(mask));
    end
end 

##
# t1 = heatmap(abs.(Map[:,:,1]), color=:jet, title="Linear Spaced");
# t2 = heatmap(abs.(Map[:,:,2]), color=:jet, title="Log Spaced");
# t3 = heatmap(abs.(Map[:,:,3]), color=:jet, title="CRLB");
# t4 = heatmap(abs.(Map[:,:,4]), color=:jet, title="MCRLB");
# t5 = heatmap(abs.(Map[:,:,5]), color=:jet, title="Ground Truth");
# t6 = heatmap(abs.(Map[:,:,6]), color=:jet);
# t7 = heatmap(abs.(Map[:,:,7]), color=:jet);
# t8 = heatmap(abs.(Map[:,:,8]), color=:jet);
# t9 = heatmap(abs.(Map[:,:,9]), color=:jet);
# t10 = heatmap(abs.(Map[:,:,10]), color=:jet);

cbs = permutedims([mod(i,C)==0 for i in 1:P*C]);
titles = [i <= C ? crit_titles[i] : "" for i in 1:P*C];
ylabels = [mod(i-1,C) == 0 ? parameter_titles[1+div(i,C)] : "" for i in 1:P*C];

img = plot([heatmap(abs.(Map[:,:,i]), color=:jet, title=titles[i], ylabel=ylabels[i], yguidefontrotation = -90) for i in 1:P*C]...,
    layout=(P,C), yflip=true, ticks=nothing, 
    colorbar=cbs, margin=10px, size=[1600,260*P+100])
savefig(img, "images//$(model).png")