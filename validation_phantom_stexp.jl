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
using MATLAB
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

function get_filename(mod, crit)
    return "../CAI2R_T1rho_phantom\\data\\Syn_noise\\images\\rec_images_$(mod)_$(crit)1.mat"
end

model = "stexp";
crits = ["lin", "log", "crlb", "mcrlb"];

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

    bds = [0.0 20 0.1; 1e10 70 1.0];
    w = [0; 1; 1];
    tnrmse = zeros(N);
    tmnae = zeros(N);
    p̂ = zeros(ComplexF64, (3,N));

    G = abs.(fimgs[:, 1]);
    # fimgs ./= G;
    @time @threads for i in findall(>(0),fmask)
        p̂[:, i] = nls_trcg(fimgs[i, :], tsl, [1.0+0im; 45; 0.5], bds, "stexp", "T1rho")[1];
    end

    Amap = data["Amap"];
    Tmap = data["Tmap"];
    Bmap = data["Bmap"];
    mask = data["mask"];

    pAmap = reshape((p̂[1,:]), size(mask));
    pTmap = reshape(abs.(p̂[2,:]), size(mask));
    pBmap = reshape(abs.(p̂[3,:]), size(mask));

    # gui = imshow_gui((256,256), (3,3))
    # canvases = gui["canvas"];
    # imshow(canvases[1, 1], abs.(Amap));
    # imshow(canvases[1, 2], abs.(pAmap));
    # imshow(canvases[1, 3], abs.(abs.(Amap)-abs.(pAmap))./abs.(Amap));
    # imshow(canvases[2, 1], Tmap);
    # imshow(canvases[2, 2], pTmap);
    # imshow(canvases[2, 3], abs.(Tmap-pTmap)./abs.(Tmap));
    # imshow(canvases[3, 1], Bmap);
    # imshow(canvases[3, 2], pBmap);
    # imshow(canvases[3, 3], abs.(Bmap-pBmap)./abs.(Bmap));
    # Gtk.showall(gui["window"]);

    write_matfile("../CAI2R_T1rho_phantom\\data\\Syn_noise\\images\\rec_params_$(model)_$(crit).mat";
    pAmap = pAmap, pTmap = pTmap, pBmap = pBmap, Amap = Amap, Bmap = Bmap, Tmap = Tmap)
end