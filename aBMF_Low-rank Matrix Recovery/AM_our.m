function [hist_tt, hist_gamma, hist_penalty, hist_testError, hist_obj] = AM_our(opts)
tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A0    = opts.A0;
Q_l    = opts.Q_l;
Q_r    = opts.Q_r;
trainY = opts.trainY;
testY  = opts.testY;
sEE    = opts.sEE;
eta   = opts.eta;
X     = opts.X0; 
Y     = opts.Y0; 
maxit = opts.maxit;
gamma = opts.gamma0;
tol   = opts.tol;
back  = opts.back;
clear opts;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[N,dim] = size(X);  E_dim = eye(dim);
[rows, cols, labs] = find(A0);
MCL_1_2 = [rows,cols];  
idxs_c = unique(MCL_1_2);  
idxs_u = setdiff(unique((1:N)'),idxs_c); 
nc = length(idxs_c);  
idxs_m_with = cell(N,1); labs_m_with = cell(N,1); 
for p = 1:nc
    m = idxs_c(p);
    [r_m,c_m,~] = find(MCL_1_2==m);
    idx_tmp = MCL_1_2(sub2ind(size(MCL_1_2), r_m, 3-c_m));
    idxs_m_with{m}  = idx_tmp; 
    labs_m_with{m} = labs(r_m,:);  
end
clear MCL_1_2 tmp_labs;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:maxit    
    gl = gamma + eta;
    for p = 1:nc  
        m  = idxs_c(p);
        labs_m = labs_m_with{m};        
        Vm = Y(idxs_m_with{m},:);        
        bm = sum(diag(labs_m)*Vm,1) + gl*Y(m,:); 
        X(m,:) = bm/(gl*E_dim + Vm'*Vm);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Vm = X(idxs_m_with{m},:);
        bm = sum(diag(labs_m)*Vm,1) + gl*X(m,:);
        Y(m,:) = bm/(gl*E_dim + Vm'*Vm);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    end
    X(idxs_u,:) = (gamma/gl)*Y(idxs_u,:);
    Y(idxs_u,:) = (gamma/gl)*X(idxs_u,:);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    QlY   = Q_l*Y; QrY = Q_r*Y; QlX = Q_l*X; 
    resxy = (sum(bsxfun(@times, QlY, QrY),2)) - labs;  
    Gz    = Q_l'*spdiags(resxy,0,sEE)*Q_r; 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    hist_obj(k,1)     = fval_mc(Y, QlY, QrY, labs, eta); 
    hist_penalty(k,1) = sqrt(sum(sum((X-Y).*(X-Y))));
    hist_gamma(k,1)   = gamma;    
    hist_testError(k,1) = RMSE((X+Y)/2, trainY, testY);
    hist_tt(k,1)      = toc;    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(k < back + 1)
        obj_old = 0;
    else
        obj_old = hist_obj(k-back);
    end
    if(abs(hist_obj(k) - obj_old)/max(hist_obj(k), 1) < tol)
        break;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    gamma = AM_our_adp(X-Y, Gz, eta, gamma);  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
end
end

function [gamma] = AM_our_adp(dXY, Gz, eta, gamma_old)
ak  = 0.3; 
fz = sum(sum(dXY.*(Gz*dXY)));
Qz = fz/sum(sum(dXY.*dXY))/ak;
gam_check = max(Qz/2 - eta/4, 0) + 1e-5; 

Gzs = abs(Gz);
nm1 = sqrt(max(sum(Gzs,1)));
nmi = sqrt(max(sum(Gzs,2)));
gam_hat = sqrt(nm1*nmi)/2 - eta/4;
gamma  = max(min(gam_hat, gamma_old), gam_check);
end

function [testError] = RMSE(V, trainY, testY)

[m, n] = size(trainY);
EE   = speye(m+n,m+n); 
markTestY = (testY ~= 0);
[rowTestY, colTestY] = find(markTestY);
Testlabs = testY(markTestY);

Q_l  = EE(rowTestY,:);
Q_r  = EE(m+colTestY,:);

tmp = (sum(bsxfun(@times, Q_l*V, Q_r*V),2)) - Testlabs;
testError = sqrt(sum(tmp.*tmp)/length(Testlabs));
end

