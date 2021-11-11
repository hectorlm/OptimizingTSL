include("TRCG_NLS.jl")
using Base.Threads
using Plots
using Statistics

TSLs = Dict();
# TSLs[1] = [1;53];
# TSLs[2] = [1;55;55];
# TSLs[3] = [1;55;55;55];
# TSLs[4] = [1;55;55;55;55];
# TSLs[5] = [1;55;55;55;55;55];
# TSLs[6] = [1;55;55;55;55;55;55];
# TSLs[7] = [1;55;55;55;55;55;55;55];
# TSLs[8] = [1;55;55;55;55;55;55;55;55];
TSLs[1] = [1.;7;55];
TSLs[2] = [1.;7;7;55];
TSLs[3] = [1.;7;7;8;55];
TSLs[4] = [1.;1;6;6;7;55];
TSLs[5] = [1.;1;7;7;7;55;55];
TSLs[6] = [1.;1;7;7;7;8;55;55];
TSLs[7] = [1.;1;7;7;7;7;8;55;55];
TSLs[8] = [1.;1;1;6;6;7;7;7;55;55];

L = length(TSLs);

forward = (p::Matrix{ComplexF64}, TSLs::Vector{Float64}) -> p[1, :].*exp.(-(TSLs'./p[2, :]).^p[3, :]);


bds = [0.0 20 0.1; 1e10 70 1.0];
w = [0; 1; 1];
mse = zeros(L);
mnae = zeros(L);
N = 16*300;
SNR = 30;
# Cost = Dict();
p̂ = zeros(Complex, (3,N));
p = zeros(Complex, (3,N))
n=1;
for n in 1:L
    global Cost, p̂, p
    A = randn(ComplexF64, N);
    A ./= abs.(A);
    T1ρ = rand(Set(20:0.1:70), N);
    Betas = rand(Set(0.1:0.01:1.0), N);
    p = permutedims([A T1ρ Betas]);

    x = forward(p, TSLs[n]);
    y = x + (1/SNR)*randn(ComplexF64, size(x));
    G = abs.(y[:, 1]);
    y ./= G;

    @time @threads for i in 1:N
        p̂[:, i] = nls_trcg(y[i, :], TSLs[n], [1.0+0im; 45; 0.8], bds, "stexp", "T1rho")[1];
    end

    p̂[1, :] .*= G;
    # println()
    ae = sum(w.*(mean(abs.(p-p̂)./abs.(p), dims=2)))/(sum(w));
    mnae[n] = ae;
    mse[n] = sum(w.*(sqrt.(mean(abs.(p-p̂).^2,dims=2)./(mean(abs.(p).^2, dims=2)))))/sum(w);
    display(mnae[n])
end
plot(mnae)
plot(mse)