using XLSX
using JLD2
using Base: @kwdef

@kwdef mutable struct tsl_params
    model::String
    N_TSLs::Int
    opt_type::String
    opt_crit::String
    model_var::String
    SNR::Int
    TSLs::Matrix{Float64}
end

model = "biexp"; # "mono", "stexp", "biexp"

if model == "mono"
    msheet = "Monoexp";
elseif model == "stexp"
    msheet = "Stexp";
else
    msheet = "Biexp";
end

tab = load("data/biexp_models_nonopt_alt.jld2")
entries = length(tab["ModelParams"]);
tab2 = load("data/bi_models_alt_100000.jld2")
entries2 = length(tab2["ModelParams"]);
XLSX.openxlsx("results/summary_new.xlsx", mode="rw") do xf
    off = offset = 0;
    sheet = xf[msheet];
    for n in 2:entries+1
        sheet["A$(n+offset)"] = tab["ModelParams"][n-1].N_TSLs;
        sheet["C$(n+offset)"] = tab["ModelParams"][n-1].opt_crit;
        sheet["B$(n+offset)"] = tab["ModelParams"][n-1].opt_type;
        sheet["D$(n+offset)"] = tab["ModelParams"][n-1].model_var;
        sheet["E$(n+offset)"] = tab["ModelParams"][n-1].SNR;
        sheet["F$(n+offset)"] = "$(tab["ModelParams"][n-1].TSLs)";
        sheet["G$(n+offset)"] = tab["MNAE"][n-1];
        sheet["H$(n+offset)"] = tab["NRMSE"][n-1];
        off += 1;
    end
    offset = off;
    for n in 2:entries2+1
        sheet["A$(n+offset)"] = tab2["ModelParams"][n-1].N_TSLs;
        sheet["C$(n+offset)"] = tab2["ModelParams"][n-1].opt_crit;
        sheet["B$(n+offset)"] = tab2["ModelParams"][n-1].opt_type;
        sheet["D$(n+offset)"] = tab2["ModelParams"][n-1].model_var;
        sheet["E$(n+offset)"] = tab2["ModelParams"][n-1].SNR;
        sheet["F$(n+offset)"] = "$(tab2["ModelParams"][n-1].TSLs)";
        sheet["G$(n+offset)"] = tab2["MNAE"][n-1];
        sheet["H$(n+offset)"] = tab2["NRMSE"][n-1];
        off += 1;
    end
end
