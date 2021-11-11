include("./ann_fit.jl")
using .ann_fit
using .ann_fit: def_model_bi, getdata_bi, gendata_bi, Model_Args
using Flux
using Plots
using Statistics
using MAT
using Base: @kwdef
using LightXML
using Printf: @sprintf
using JLD2
plotlyjs()

function nmae(pred, truevalue, weights=1)
    error = abs.(truevalue-pred)./abs.(truevalue);
    if weights == 1
        error = mean(error)
    else
        error = mean(weights.*mean(error, dims=2))
    end
    return error
end

function nrmse(pred, truevalue, weights=1)
    error = mean(abs.(truevalue-pred).^2,dims=2);
    if weights == 1
        error = mean(sqrt.(error./mean(truevalue.^2, dims=2)))
    else
        error = mean(weights.*sqrt.(error./mean(truevalue.^2,dims=2)))
    end
    return error
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

function get_filename(param, TCRLB, N_TSL, SNR, crit, mod)
    return "..//TSL_opt_Hector//sampling_times//TSLs_$(param)_$(TCRLB)_N$(N_TSL)_SNR$(SNR)_$(crit)_$(mod).mat"
end

function get_param_xml(filepath)
    xdoc = parse_file(filepath);
    xroot = root(xdoc);
    SNR = parse.(Int, content.(xroot["SNR"]));
    N_TSLs = parse.(Int, content.(xroot["TSLs"]));
    mcrlb = content.(xroot["mcrlb"]);
    crit = content.(xroot["crit"]);
    time_rate = content.(xroot["time_rate"]);
    free(xdoc)
    return SNR, N_TSLs, mcrlb, crit, time_rate
end

SNR, N_TSLs, mcrlb, crit, time_rate = get_param_xml(".//params.xml")

# SNR = [30];
# N_TSLs = [4];
# mcrlb = ["MCRLB"];
# crit = ["mean" "max"];
# time_rate = ["T1rho"];

param_args = Vector{tsl_params}();
for s in SNR
    for n in N_TSLs
        for m in mcrlb # CRLB or MCRLB
            for c in crit # Mean or Max Error
                for r in time_rate # T1rho or R1rho
                    # for mod in ["monoexp" "biexp" "stexp"]
                    for mod in ["biexp"]
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

trials = 15;

tab = Dict();
R2i = zeros(length(param_args));
model_loss = zeros(length(param_args));
model_nrmse = zeros(length(param_args));
model_nmae = zeros(length(param_args));
tab = load("bi_models.jld2");
R2i = tab["R2"];
model_loss = tab["Loss"];
model_nrmse = tab["NRMSE"];
model_nmae = tab["MNAE"];
# try
#     global tab = load("bi_models.jld2");
#     global R2i = tab["R2"];
#     global model_loss = tab["Loss"];
#     global model_nrmse = tab["NRMSE"];
#     global model_nmae = tab["MNAE"];
# catch ArgumentError
#     println("File not found!")
# end

weights = [0 0 1 1 1]';
for inst in 20:length(param_args)
    @info "Instance $inst"
    display(param_args[inst])
    print("\n")
    args = Model_Args(;n_epochs=4000, TSL=param_args[inst].TSLs, SNR=param_args[inst].SNR, debug=true, lw=96)
    R2t = zeros(3, trials);
    loss_it = zeros(trials);
    t_nrmse = zeros(trials);
    t_nmae = zeros(trials);
    for trial in 1:trials
        print("\r Trial $trial")
        model = def_model_bi(args.N_TSL, args.lw);
        dataloaders = getdata_bi(args);
        # @time model = train(model, dataloaders, args)
        model = train(model, dataloaders, args)

        Flux.testmode!(model)
        x,y = gendata_bi(args, 5000)
        predy = model(x)
        e = predy[3:end,:]-y[3:end,:];
        ssres = sum(e.^2, dims=2);
        sstot = sum((y[3:end,:].-mean(y[3:end,:],dims=2)).^2,dims=2);
        R2t[:, trial] = 1 .- ssres./sstot;
        loss_it[trial] = mean((predy-y)[3,:].^2 ./ abs.(y[3,:]));
        t_nrmse[trial] = nrmse(predy, y, weights);
        t_nmae[trial] = nmae(predy, y, weights);
        # display("Test loss = $(test_loss[trial])")
        # display("R-Squared: $(R2t[trial])")
        # display(scatter(y[3,:],predy[3,:],xlabel="y",ylabel="pred_y", smooth=true))
    end
    R2i[inst] = mean(R2t);
    model_loss[inst] = mean(loss_it);
    model_nrmse[inst] = mean(t_nrmse)*100;
    model_nmae[inst] = mean(t_nmae)*100;
    println()
    @info "R2 = $(R2i[inst])"
    @info "Model loss = $(model_loss[inst])"
    @info "Model NRMSE = $(@sprintf("%.2f", model_nrmse[inst]))%"
    @info "Model MNAE = $(@sprintf("%.2f", model_nmae[inst]))%"
    println(repeat("=",182))
end

tab = Dict("ModelParams" => param_args, "NRMSE"=>model_nrmse,
             "MNAE"=>model_nmae, "Loss"=>model_loss, "R2"=>R2i);
save("bi_models.jld2", tab)