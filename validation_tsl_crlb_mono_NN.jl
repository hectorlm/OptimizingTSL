include("./ann_fit.jl")
# using .ann_fit
using Flux
using Plots
using Statistics
using MAT
using Base: @kwdef
using LightXML
using Printf: @sprintf
using JLD2
plotlyjs()
# include("ann_fit.jl")
# using .ann_fit

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

# function get_param_xml(filepath)
#     xdoc = parse_file(filepath);
#     xroot = root(xdoc);
#     SNR = parse.(Int, content.(xroot["SNR"]));
#     N_TSLs = parse.(Int, content.(xroot["TSLs"]));
#     mcrlb = content.(xroot["mcrlb"]);
#     crit = content.(xroot["crit"]);
#     time_rate = content.(xroot["time_rate"]);
#     free(xdoc)
#     return SNR, N_TSLs, mcrlb, crit, time_rate
# end

# SNR, N_TSLs, mcrlb, crit, time_rate = get_param_xml(".//params.xml")

SNR = [125];
N_TSLs = [4 5];
mcrlb = ["CRLB"];
crit = ["mean"];
time_rate = ["T1rho"];

param_args = Vector{tsl_params}();
for s in SNR
    for n in N_TSLs
        for m in mcrlb # CRLB or MCRLB
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

tab = Dict();
R2i = zeros(length(param_args));
model_loss = zeros(length(param_args));
model_nrmse = zeros(length(param_args));
model_nmae = zeros(length(param_args));

# try
#     global tab = load("st_models.jld2");
#     global R2i = tab["R2"];
#     global model_loss = tab["Loss"];
#     global model_nrmse = tab["NRMSE"];
#     global model_nmae = tab["MNAE"];
# catch ArgumentError
#     pass;
# end

weights = [0 0 1]';
model = [];
for inst in 1:length(param_args)
    @info "Instance $inst"
    display(param_args[inst])
    print("\n")
    train_args = Model_Args(; t1ρ_l=10, t1ρ_h=100, n_epochs=2000,TSL=param_args[inst].TSLs, SNR=param_args[inst].SNR, debug=true, len_train=30000, batchsize=30000)
    test_args = Model_Args(; TSL=param_args[inst].TSLs, SNR=param_args[inst].SNR)
    R2t = zeros(trials);
    loss_it = zeros(trials);
    t_nrmse = zeros(trials);
    t_nmae = zeros(trials);
    for trial in 1:trials
        global model
        print("\r Trial $trial")
        model = def_model_mono(train_args.N_TSL, train_args.lw);
        dataloaders = getdata_mono(train_args);
        # @time model = train(model, dataloaders, args)
        model = train(model, dataloaders, train_args)

        # Flux.testmode!(model)
        x,y = gendata_mono(test_args, 30000)
        predy = model(x)
        e = predy[3,:]-y[3,:];
        ssres = sum(e.^2);
        sstot = sum((y[3,:].-mean(y[3,:])).^2)
        R2t[trial] = 1 - ssres/sstot;
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
    @info "Model NMAE = $(@sprintf("%.2f", model_nmae[inst]))%"
    println(repeat("=",182))
end

tab = Dict("ModelParams" => param_args, "NRMSE"=>model_nrmse,
             "NMAE"=>model_nmae, "Loss"=>model_loss, "R2"=>R2i);
# save("mono_models.jld2", tab)