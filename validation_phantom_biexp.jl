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
using ImageView
using Gtk.ShortNames
# plotlyjs()
using MATLAB

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
    return "../CAI2R_T1rho_phantom\\data\\Syn_noise\\images\\rec_images_$(mod)_$(crit)1.mat"
end

model = "biexp";
crits = ["lin", "log", "crlb", "mcrlb"];
# crits = ["mcrlb"];

for crit in crits
    st = get_filename(model, crit);
    file = matopen(st);
    data = read(file);
    close(file);

    tsl = data["TSL"][:];
    N_TSL = length(tsl);
    fimgs = reshape(data["recim"], (:,N_TSL));
    fmask = data["mask"][:];
    N = size(fimgs, 1);

    bds = [0.0 0.05 15 0.1; 1e10 0.95 100 10];
    w = [0; 1; 1; 1];
    tnrmse = zeros(N);
    tmnae = zeros(N);
    p̂ = zeros(ComplexF64, (4,N));

    G = abs.(fimgs[:, 1]);
    fimgs ./= G;
    @time @threads for i in findall(>(0),fmask)
        p̂[:, i] = nls_trcg(fimgs[i, :], tsl, [1.0+0im; 0.5; 55; 5.5], bds, "biexp", "T1rho")[1];
    end

    Amap = data["Amap"];
    Tsmap = data["Tsmap"];
    Tmap = data["Tmap"];
    Fmap = data["Fmap"];

    pAmap = reshape((p̂[1,:]).*G, size(data["mask"]));
    pFmap = reshape(real.(p̂[2,:]), size(data["mask"]));
    pTmap = reshape(real.(p̂[3,:]), size(data["mask"]));
    pTsmap = reshape(real.(p̂[4,:]), size(data["mask"]));

    # gui = imshow_gui((256,256), (4,3));
    # canvases = gui["canvas"];
    # imshow(canvases[1, 1], abs.(Amap));
    # imshow(canvases[1, 2], abs.(pAmap));
    # imshow(canvases[1, 3], abs.(Amap-pAmap)./abs.(Amap));
    # imshow(canvases[2, 1], abs.(Fmap));
    # imshow(canvases[2, 2], abs.(pFmap));
    # imshow(canvases[2, 3], abs.(Fmap-pFmap)./abs.(Fmap));
    # imshow(canvases[3, 1], Tmap);
    # imshow(canvases[3, 2], pTmap);
    # imshow(canvases[3, 3], abs.(Tmap-pTmap)./abs.(Tmap));
    # imshow(canvases[4, 1], Tsmap);
    # imshow(canvases[4, 2], pTsmap);
    # imshow(canvases[4, 3], abs.(Tsmap-pTsmap)./abs.(Tsmap));
    # Gtk.showall(gui["window"]);
    write_matfile("../CAI2R_T1rho_phantom\\data\\Syn_noise\\images\\rec_params_$(model)_$(crit).mat";
    pAmap = pAmap, pFmap = pFmap, pTmap = pTmap, pTsmap = pTsmap, Amap = Amap, Tsmap = Tsmap, Tmap = Tmap, Fmap = Fmap)
end
