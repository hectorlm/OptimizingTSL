# __precompile__()

# module ann_fit

# export Model_Args, gendata_mono, def_model_mono, train, getdata_mono, gendata_bi, def_model_bi, getdata_bi, getdata_stretch, gendata_stretch, def_model_stretch

using Flux
using CUDA
using Statistics
# using Plots
using LinearAlgebra: norm
using Base: @kwdef
# plotlyjs()

@kwdef mutable struct Model_Args
    η::Float64 = 2.5e-3               # learning rate
    n_epochs::Int = 3000            # number of epochs to train
    use_cuda::Bool = true           # run on gpu
    t1ρ_l::Float64 = 20             # lower range of t1ρ for data generation
    t1ρ_h::Float64 = 70             # upper range
    t1ρs_l::Float64 = 1             # Lower bound for short t1ρ range
    t1ρs_h::Float64 = 10            # Upper bound for short t1ρ range
    fraction_l::Float64 = 0.2       # Lower bound for fraction range
    fraction_h::Float64 = 0.8       # Upper bound for fraction range   
    β::Float64 = 0.1       # Lower bound for stretching parameter (Upper bound is 1.0) 
    SNR::Int = 30                   # SNR used in data generation
    TSL::Array{Float64} = [1 45]    # TSLs used in data generation
    N_TSL = length(TSL)             # how many TSLs
    len_train::Int = 20000          # how many repetitions of the t1ρ_range used in training
    batchsize::Int = len_train      # batch size
    len_test::Int = 5000            # how many repetitions of the t1ρ_range used in testing
    lw = 12                    # Hidden Layer width
    debug::Bool = false
end

function gendata_mono(args, N)
    A = randn(ComplexF64, (N,1));
    A ./= abs.(A);
    T = rand(Set(args.t1ρ_l:0.5:args.t1ρ_h), (N,1));#t1ρ_range[1].+(t1ρ_range[end]-t1ρ_range[1]).*rand(N,1);
    target = permutedims([real(A) imag(A) (1 ./ T)]);
    noise = 1/(args.SNR);
    f(x) = ((x[1, :]+x[2, :]im).*exp.(-args.TSL.*x[3, :]));
    data = f(target);
    data += noise.*randn(ComplexF64, size(data));
    G = abs.(data[:,1]);
    data = permutedims(Array([abs.(data)./G angle.(data)]));
    target = permutedims([real(A)./G imag(A)./G (1 ./ T)]);
    return (data, target)
end

function gendata_bi(args, N)
    A = randn(ComplexF64, (N,1));
    A ./= abs.(A);
    Tl = rand(Set(args.t1ρ_l:0.5:args.t1ρ_h), (N,1))
    Ts = rand(Set(args.t1ρs_l:0.5:args.t1ρs_h), (N,1))
    F = rand(Set(args.fraction_l:0.01:args.fraction_h), (N,1))
    target = permutedims([real(A) imag(A) F (1 ./ Tl) (1 ./ Ts)]);
    noise = 1/(args.SNR);
    f(x) = (x[1, :]+x[2, :]im).*(x[3,:].*exp.(-args.TSL.*x[4, :]) + (1 .-x[3,:]).*exp.(-args.TSL.*x[5,:]));
    data = f(target);
    data += noise.*randn(ComplexF64, size(data));
    G = abs.(data[:,1]);
    data = permutedims([abs.(data)./G; angle.(data)]);
    target[1:2, :] = permutedims([real(A)./G imag(A)./G]);
    return (data, target)
end

function gendata_stretch(args, N)
    A = randn(ComplexF64, (N,1));
    A ./= abs.(A);
    T = rand(Set(args.t1ρ_l:0.5:args.t1ρ_h), (N,1))
    Β = rand(Set(args.β:0.1:1.0), (N,1))
    target = permutedims([real(A) imag(A) (1 ./ T) Β]);
    noise = 1/(args.SNR);
    f(x) = (x[1, :]+x[2, :]im).*exp.(-(args.TSL.*x[3, :]).^x[4,:]);
    data = f(target);
    data += noise.*randn(ComplexF64, size(data));
    G = abs.(data[:,1]);
    data = permutedims([abs.(data)./G; angle.(data)]);
    target[1:2, :] = permutedims([real(A)./G imag(A)./G]);
    return (data, target)
end

function getdata_mono(args)
    data_train, target_train = gendata_mono(args, args.len_train);
    data_test, target_test = gendata_mono(args, args.len_test);

    train_loader = Flux.DataLoader((data=data_train, target=target_train), batchsize=args.batchsize, shuffle=true);
    test_loader = Flux.DataLoader((data_test, target_test), batchsize=args.len_test);
    return train_loader, test_loader
end

function getdata_bi(args)
    data_train, target_train = gendata_bi(args, args.len_train);
    data_test, target_test = gendata_bi(args, args.len_test);

    train_loader = Flux.DataLoader((data=data_train, target=target_train), batchsize=args.batchsize, shuffle=true);
    test_loader = Flux.DataLoader((data_test, target_test), batchsize=args.len_test);
    return train_loader, test_loader
end

function getdata_stretch(args)
    data_train, target_train = gendata_stretch(args, args.len_train);
    data_test, target_test = gendata_stretch(args, args.len_test);

    train_loader = Flux.DataLoader((data=data_train, target=target_train), batchsize=args.batchsize, shuffle=true);
    test_loader = Flux.DataLoader((data_test, target_test), batchsize=args.len_test);
    return train_loader, test_loader
end

function def_model_mono(N_TSL, hl_size)
    model = Chain(
        Dense(2*N_TSL,128,gelu),
        Dense(128, 3),
        Dense(3, 128, gelu),
        Dense(128,3)
    );
    return model
end

function def_model_bi(N_TSL, hl_size)
    model = Chain(
        Dense(2*N_TSL, hl_size, gelu),
        Dense(hl_size, 5)
    )
    return model
end

function def_model_stretch(N_TSL, hl_size)
    model = Chain(
        Dense(2*N_TSL, hl_size, gelu),
        Dense(hl_size, 4)
    )
    return model
end

function loss(dataloader, model, device)
    ls = 0.00f0
    num = 0
    for (x, y) in dataloader
        x, y = device(x), device(y)
        ls += Flux.Losses.mse(model(x), y, agg=mean)
        num += size(x, 2)
    end
    return ls#/num
end

function custom_loss(yh, y)
    error = abs.(yh-y);
    return norm(error, Inf);
end

function train(model, dataloader, args)
    # args = Model_Args(; kws...)

    if CUDA.functional() && args.use_cuda
        # if args.debug
        #     @info "Training on CUDA GPU"
        # end
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    # train_loader, test_loader = getdata_mono(args, device)
    train_loader, test_loader = dataloader

    # model = def_model(args.N_TSL, args.lw) |> device;
    model = model |> device;
    # opt = Flux.Optimise.Momentum(args.η);
    opt = Flux.Optimise.ADAM(args.η);
    pm = params(model);

    test_loss = zeros(args.n_epochs);
    train_loss = zeros(args.n_epochs);
    for e in 1:args.n_epochs
        for (x, y) in train_loader
            x, y = device(x), device(y);
            gs = gradient(()-> Flux.Losses.mse(model(x), y), pm);
            # gs = gradient(()-> custom_loss(model(x), y), pm);
            Flux.Optimise.update!(opt, pm, gs);
        end
        train_loss[e] = loss(train_loader, model, device)
        if args.debug
            test_loss[e] = loss(test_loader, model, device);
            if (e-1)%50==0
                print("\rEpoch: $e - Train loss: $(train_loss[e])\t")
                print("Test loss: $(test_loss[e])")
            end
        end
    end
    if args.debug
        a = plot(train_loss, xlabel="epochs", ylabel="Train Error", yscale=:log10)
        b = plot(test_loss, xlabel="epochs", ylabel="Test Error", yscale=:log10)
        display(plot(a,b))
    end
    return cpu(model)
end

# end
