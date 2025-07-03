# module TRCG_NLS

# export nls_trcg

using LinearAlgebra: norm


function pocs(x::Vector{ComplexF64}, bounds::Matrix{Float64})
    @assert(length(x)==size(bounds,2))
    # px = similar(x);
    x[1] = sign.(@view x[1]).*min.(max.(abs.(x[1]), bounds[1,1]), bounds[2,1]);
    x[2:end] = min.(max.(real.(@view x[2:end]), @view bounds[1,2:end]),@view bounds[2,2:end]);
    return x
end


function nls_trcg(b::Vector{ComplexF64}, TSLs::Vector{Float64}, p0::Vector{ComplexF64}, bounds::Matrix{Float64}, model::String, model_var::String; maxiter=10000, refsse=1e200)
    errorTol = 1e-9;
    cgtol = 1e-17;
    p = p0;
    p .= pocs(p, bounds);
    Cost = zeros(Float64, (maxiter));
    p_diff = zeros(Float64, (maxiter));
    delta_k = zeros(Float64, (maxiter));
    if model == "monoexp"
        if model_var == "R1rho"
            f, fJ = fw_mo, JmoR
        else
            f, fJ = fw_mo, JmoT
        end
    elseif model == "stexp"
        if model_var == "R1rho"
            f, fJ = fw_st, JstR
        else
            f, fJ = fw_st, JstT
        end
    else
        if model_var == "R1rho"
            f, fJ = fw_bi, JbiR
        else
            f, fJ = fw_bi, JbiT
        end
    end
    fw = f(p, TSLs)[:];
    res = b - fw;
    J = fJ(p, TSLs);
    g = J'*res;
    z = zeros(ComplexF64, size(g));
    k=1;
    L2cost = sum(abs.(res).^2);
    Cost[k] = L2cost;
    
    DOF = length(p) + 3;

    costdown = true;
    costdown_old = true;
    stopping = false;
    delta = 1e-1;
    # delta = 1e-12;
    p_diff[k] = errorTol+eps();
    # iterations
    while( ~stopping && (k<maxiter))
        p_old = p;
        res_old = res;
        L2cost_old = L2cost;
        J .= fJ(p, TSLs);

        g = J'*res;
        A = J'J;

        # CG limited to the trust region
        d = g;
        gg = g'*g;
        z = zeros(ComplexF64, size(g));
        CGiter = true;
        j=1;
        while(CGiter)
            if(abs(gg) < cgtol)
                h=z;
                CGiter = false;
            end
            Ad = A*d;

            alpha = gg/abs.(d'*Ad);
            zn = z + alpha*d;
            n_zn = norm(zn);

            if(n_zn>delta)
                zz = abs.(z'*z);
                dd = abs.(d'*d);
                zd = real.(z'*d);
                bb = (-2*zd+sqrt(4*abs.(zd).^2 - 4*(zz-delta^2)*dd))/(2*dd);
                h = z + bb*d;
                choice = j;
                CGiter = false;
            end

            z = zn;
            g -= alpha*Ad;
            gg_old = gg;

            gg = g'*g;

            bb = gg/gg_old;
            d = g + bb*d;
            choice = j;
            j += 1;

            if(j > DOF)
                h = z;
                CGiter = false;
            end
        end

        z = p + h;

        # projected interior point

        pp = z;
        pp = pocs(pp, bounds);

        fwp  = f(pp, TSLs)[:];
        res = b - fwp;

        L2cost = sum(abs.(res).^2);
        p = pp;

        h = p - p_old;
        pred = 1 .*(-g'*h + .5*h'*A*h);

        costdown_old = costdown;
        costdown = L2cost < (L2cost_old + 10*eps());

        rho = (L2cost_old - L2cost)/(abs(pred)+eps());
        M = 10;
        if rho<.25
            delta /= 1.5*M;
        elseif rho>.75
            delta *= 1.0*M;
        end
        delta_k[k] = delta;
        if ~costdown
            L2cost = L2cost_old;
            p = p_old;
            res = res_old;
        end
        k += 1;

        Cost[k] = L2cost;
        p_diff[k] = norm(p-p_old,2)/norm(p);
        ki = max(1, k-100);
        if ((k>100) && (~costdown)==0 && (sum(p_diff[ki:k])<errorTol) && (L2cost<=refsse))
            stopping = true;
            # Cost[k+1:end] .= Cost[k];
        end
    end
    return p, Cost, p_diff, delta_k
end

function fw_mo(p::Vector{ComplexF64}, TSLs::Vector{Float64})
    return p[1, :].*exp.(-(TSLs'./p[2, :]))
end

function JmoR(p::Vector{ComplexF64}, TSLs::Vector{Float64})
    return permutedims(
            [
                exp.(-(TSLs'./p[2]))
                -p[1].*TSLs'.*exp.(-(TSLs'./p[2])) 
            ]);
end

function JmoT(p::Vector{ComplexF64}, TSLs::Vector{Float64})
    return permutedims(
            [
                exp.(-(TSLs'./p[2]))
                p[1]/(p[2]^2).*TSLs'.*exp.(-(TSLs'./p[2]))
            ]);
end

function fw_st(p::Vector{ComplexF64}, TSLs::Vector{Float64})
    return p[1, :].*exp.(-(TSLs'./p[2, :]).^p[3, :]);
end

function JstT(p::Vector{ComplexF64}, TSLs::Vector{Float64})
    return permutedims(
            [
                exp.(-(TSLs'./p[2]).^p[3])
                p[1]*p[3]./p[2]*exp.(-(TSLs'./p[2]).^p[3]).*((TSLs'./p[2]).^p[3])
                -p[1]*exp.(-(TSLs'./p[2]).^p[3]).*((TSLs'./p[2]).^p[3]).*log.(TSLs'./p[2])
            ]);
end
function JstR(p::Vector{ComplexF64}, TSLs::Vector{Float64})
    return permutedims(
            [
                exp.(-(TSLs'./p[2]).^p[3])
                -p[1]*p[3]/p[2]*exp.(-(TSLs'/p[2]).^p[3]).*((TSLs'/p[2]).^p[3])
                -p[1]*exp.(-(TSLs'./p[2]).^p[3]).*((TSLs'./p[2]).^p[3]).*log.(TSLs'./p[2])
                ]);
end

function fw_bi(p::Vector{ComplexF64}, TSLs::Vector{Float64})
    return p[1].*(p[2].*exp.(-(TSLs'./p[3])) + (1 .-p[2]).*exp.(-(TSLs'./p[4])));
end
function JbiT(p::Vector{ComplexF64}, TSLs::Vector{Float64})
    return permutedims(
                    [
                        p[2].*exp.(-(TSLs'./p[3])) + (1-p[2]).*exp.(-(TSLs'./p[4]))
                        p[1].*(exp.(-(TSLs'./p[3])) - exp.(-(TSLs'./p[4])))
                        p[1]*p[2]/(p[3]^2).*TSLs'.*exp.(-TSLs'./p[3])
                        p[1]*(1-p[2])/(p[4]^2).*TSLs'.*exp.(-TSLs'./p[4])
                    ]);
end
function JbiR(p::Vector{ComplexF64}, TSLs::Vector{Float64})
    return permutedims(
                    [
                        p[2].*exp.(-(TSLs'./p[3])) + (1-p[2]).*exp.(-(TSLs'./p[4]))
                        p[1].*(exp.(-(TSLs'./p[3])) - exp.(-(TSLs'./p[4])))
                        -p[1]*p[2].*TSLs'.*exp.(-TSLs'./p[3])
                        -p[1]*(1-p[2]).*TSLs'.*exp.(-TSLs'./p[4])
                    ]);
end

# end