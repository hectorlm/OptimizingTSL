include("./ann_fit.jl")
using .ann_fit
using Flux
using Plots
using Statistics

args = Model_Args(;TSL=[1 45], debug=true)

model = def_model_mono(args.N_TSL, args.lw);
dataloaders = getdata_mono(args);
@time model = train(model, dataloaders, args)

Flux.testmode!(model)
x,y = gendata_mono(args, 5000)
predy = model(x)
e = predy[3,:]-y[3,:];
ssres = sum(e.^2);
sstot = sum((y[3,:].-mean(y[3,:])).^2)
R2 = 1 - ssres/sstot;
test_loss = mean((predy-y)[3,:].^2 ./ abs.(y[3,:]));
display("Test loss = $test_loss")
display("R-Squared: $R2")
scatter(y[3,:],predy[3,:],xlabel="y",ylabel="pred_y", smooth=true)