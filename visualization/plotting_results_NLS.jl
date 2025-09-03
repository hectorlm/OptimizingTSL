using XLSX
using JLD2
using Base: @kwdef
using PlotlyJS

@kwdef mutable struct tsl_params
    model::String
    N_TSLs::Int
    opt_type::String
    opt_crit::String
    model_var::String
    SNR::Int
    TSLs::Matrix{Float64}
end

tab = load("bi_models_alt_100000.jld2");
FC = tab["ModelParams"][1].N_TSLs-1; # 2 para mono, 2 para stexp, 3 para bi
snr = 30;
entries = length(tab["ModelParams"]);
if tab["ModelParams"][1].model == "biexp"
    model = "Bi-Exponential Model";
elseif tab["ModelParams"][1].model == "stexp"
    model = "Stretched Exponential Model";
    FC += 1;
else
    model = "Mono-Exponential Model";
end
tab2 = load("biexp_models_nonopt_alt.jld2");

mnae_crlb_mean_t1r = zeros(7);
mnae_crlb_mean_r1r = zeros(7);
mnae_crlb_max_t1r = zeros(7);
mnae_crlb_max_r1r = zeros(7);
mnae_mcrlb_mean_t1r = zeros(7);
mnae_mcrlb_mean_r1r = zeros(7);
mnae_mcrlb_max_t1r = zeros(7);
mnae_mcrlb_max_r1r = zeros(7);
mnae_linspaced = zeros(7);
mnae_logspaced = zeros(7);

nrmse_crlb_mean_r1r = zeros(7);
nrmse_crlb_mean_t1r = zeros(7);
nrmse_crlb_max_t1r = zeros(7);
nrmse_crlb_max_r1r = zeros(7);
nrmse_mcrlb_mean_t1r = zeros(7);
nrmse_mcrlb_mean_r1r = zeros(7);
nrmse_mcrlb_max_t1r = zeros(7);
nrmse_mcrlb_max_r1r = zeros(7);
nrmse_linspaced = zeros(7);
nrmse_logspaced = zeros(7);

for i in 1:length(tab2["ModelParams"])
    n = tab2["ModelParams"][i].N_TSLs - FC;
    if tab["ModelParams"][1].model == "stexp" && n<1
        continue
    end
    if tab2["ModelParams"][i].SNR == snr
        if tab2["ModelParams"][i].opt_crit == "Linear Spaced"
            mnae_linspaced[n] = tab2["MNAE"][i]
            nrmse_linspaced[n] = tab2["NRMSE"][i]
        else
            mnae_logspaced[n] = tab2["MNAE"][i]
            nrmse_logspaced[n] = tab2["NRMSE"][i]
        end
    end
end

for i in 1:entries
    n = tab["ModelParams"][i].N_TSLs - FC;
    if tab["ModelParams"][1].model == "stexp" && n<1
        continue
    end
    if tab["ModelParams"][i].opt_type == "CRLB"
        if tab["ModelParams"][i].opt_crit == "mean"
            if tab["ModelParams"][i].model_var == "T1rho"
                if tab["ModelParams"][i].SNR == snr
                    mnae_crlb_mean_t1r[n] = tab["MNAE"][i]
                    nrmse_crlb_mean_t1r[n] = tab["NRMSE"][i]
                end
            else
                if tab["ModelParams"][i].SNR == snr
                    mnae_crlb_mean_r1r[n] = tab["MNAE"][i]
                    nrmse_crlb_mean_r1r[n] = tab["NRMSE"][i]
                end
            end
        else
            if tab["ModelParams"][i].model_var == "T1rho"
                if tab["ModelParams"][i].SNR == snr
                    mnae_crlb_max_t1r[n] = tab["MNAE"][i]
                    nrmse_crlb_max_t1r[n] = tab["NRMSE"][i]
                end
            else
                if tab["ModelParams"][i].SNR == snr
                    mnae_crlb_max_r1r[n] = tab["MNAE"][i]
                    nrmse_crlb_max_r1r[n] = tab["NRMSE"][i]
                end
            end
        end
    else
        if tab["ModelParams"][i].opt_crit == "mean"
            if tab["ModelParams"][i].model_var == "T1rho"
                if tab["ModelParams"][i].SNR == snr
                    mnae_mcrlb_mean_t1r[n] = tab["MNAE"][i]
                    nrmse_mcrlb_mean_t1r[n] = tab["NRMSE"][i]
                end
            else
                if tab["ModelParams"][i].SNR == snr
                    mnae_mcrlb_mean_r1r[n] = tab["MNAE"][i]
                    nrmse_mcrlb_mean_r1r[n] = tab["NRMSE"][i]
                end
            end
        else
            if tab["ModelParams"][i].model_var == "T1rho"
                if tab["ModelParams"][i].SNR == snr
                    mnae_mcrlb_max_t1r[n] = tab["MNAE"][i]
                    nrmse_mcrlb_max_t1r[n] = tab["NRMSE"][i]
                end
            else
                if tab["ModelParams"][i].SNR == snr
                    mnae_mcrlb_max_r1r[n] = tab["MNAE"][i]
                    nrmse_mcrlb_max_r1r[n] = tab["NRMSE"][i]
                end
            end
        end
    end
end


x = Array(max(4, tab["ModelParams"][1].N_TSLs):tab["ModelParams"][end].N_TSLs);
traces_mnae = [
scatter(x=x, y=mnae_linspaced, name="Linear-Spaced", marker_color="black"),
scatter(x=x, y=mnae_logspaced, name="Log-Spaced", line_dash="dash", marker_color="black"),
scatter(x=x, y=mnae_crlb_mean_t1r, name="CRLB", marker_color="blue"),
scatter(x=x, y=mnae_crlb_mean_r1r, name="CRLB Mean R1ρ", marker_color="red"),
scatter(x=x, y=mnae_crlb_max_t1r, name="CRLB Maxd T1ρ", marker_color="brown"),
scatter(x=x, y=mnae_crlb_max_r1r, name="CRLB Max R1ρ", marker_color="green"),
scatter(x=x, y=mnae_mcrlb_mean_t1r, name="MCRLB", line_dash="dash", marker_color="blue"),
scatter(x=x, y=mnae_mcrlb_mean_r1r, name="MCRLB Mean R1ρ", line_dash="dash", marker_color="red"),
scatter(x=x, y=mnae_mcrlb_max_t1r, name="MCRLB Max T1ρ", line_dash="dash", marker_color="brown"),
scatter(x=x, y=mnae_mcrlb_max_r1r, name="MCRLB Max R1ρ", line_dash="dash", marker_color="green"),
];

l = Layout(title="$(model) - SNR = $(snr)dB",
                   xaxis_range=[x[1]-0.5, x[end]+0.5], xaxis_title="Number of TSLs",
                   yaxis_title="MNAE",
                   xaxis_showgrid=true, yaxis_showgrid=true)

plt_mnae_mean = plot(traces_mnae[[1,2,3,7]], l)
display(plt_mnae_mean)
# plt_mnae_max = plot(traces_mnae[[1,2,5,6,9,10]], l)
# display(plt_mnae_max)

# traces_nrmse = [
# scatter(x=x, y=nrmse_linspaced, name="Linear-Spaced", marker_color="black"),
# scatter(x=x, y=nrmse_logspaced, name="Log-Spaced", line_dash="dash", marker_color="black"),
# scatter(x=x, y=nrmse_crlb_mean_t1r, name="CRLB Mean T1ρ", marker_color="blue"),
# scatter(x=x, y=nrmse_crlb_mean_r1r, name="CRLB Mean R1ρ", marker_color="red"),
# scatter(x=x, y=nrmse_crlb_max_t1r, name="CRLB Max T1ρ", marker_color="brown"),
# scatter(x=x, y=nrmse_crlb_max_r1r, name="CRLB Max R1ρ", marker_color="green"),
# scatter(x=x, y=nrmse_mcrlb_mean_t1r, name="MCRLB Mean T1ρ", line_dash="dash", marker_color="blue"),
# scatter(x=x, y=nrmse_mcrlb_mean_r1r, name="MCRLB Mean R1ρ", line_dash="dash", marker_color="red"),
# scatter(x=x, y=nrmse_mcrlb_max_t1r, name="MCRLB Max T1ρ", line_dash="dash", marker_color="brown"),
# scatter(x=x, y=nrmse_mcrlb_max_r1r, name="MCRLB Max R1ρ", line_dash="dash", marker_color="green"),
# ];

# l = Layout(;title="$(model) - SNR = $(snr)dB",
#                    xaxis_range=[x[1]-0.5, 10.5], xaxis_title="Number of TSLs",
#                    yaxis_title="NRMSE",
#                    xaxis_showgrid=true, yaxis_showgrid=true)

# plt_nrmse = plot(traces_nrmse, l)
# display(plt_nrmse)
savefig(plt_mnae_mean, "$(model)_$(snr)dB_mnae_mean_alt.png")
# savefig(plt_mnae_max, "$(model)_$(snr)dB_mnae_max.png")