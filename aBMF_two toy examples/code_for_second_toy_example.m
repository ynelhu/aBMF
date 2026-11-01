function code_for_second_toy_example()

clear all; close all; clear;clc;
addpath(genpath(pwd)); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maxiter = 1e4;
X0 = 1; Y0 = -X0;  gamma0 = 1e-5; epsilon = 1e-3; d = 1; nu = 0.3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AM_agd %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = X0; Y = Y0; gamma = gamma0;
for i = 1:maxiter
    
    X = Y*((gamma - d)/(Y*Y + gamma));     
    Y = X*((gamma - d)/(X*X + gamma));  
    tmp = X-Y;     
        
    penal = abs(tmp);
    hist_penalty1(i,1) = penal;
    hist_obj1(i,1) = (Y*Y+d)^2/2;    
    hist_soldist1(i,1) = abs(X);    
        
    Gy = (X*Y + d)*X;
    ad = 2*sum(sum(tmp.*Gy))/sum(sum(tmp.*tmp));
    gamma = max(0, X^2 + ad) + epsilon;   
    hist_gamma1(i,1) = gamma;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AM_adp %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = X0; Y = Y0; gamma = gamma0;
for i = 1:maxiter
    X = Y*((gamma - d)/(Y*Y + gamma));  
    Y = X*((gamma - d)/(X*X + gamma));      
    tmp = X-Y;    
    
    penal = abs(tmp);
    hist_penalty2(i,1) = penal;
    hist_obj2(i,1) = (Y*Y+d)^2/2;      
    hist_soldist2(i,1) = abs(X);    
    gamma = gamma *((sum(sum(X.*X))+sum(sum(Y.*Y)))/(abs(2*sum(sum(X.*Y)))));
    hist_gamma2(i,1) = gamma;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AM_pca %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = X0; Y = Y0; gamma = gamma0;
hist_obj3     = nan(maxiter,1);
hist_gamma3   = nan(maxiter,1);
hist_soldist3 = nan(maxiter,1);
hist_penalty3 = nan(maxiter,1);
for i = 1:maxiter
    X = Y*((gamma - d)/(Y*Y + gamma));  
    Y = X*((gamma - d)/(X*X + gamma));   
    tmp = X-Y; 
    
    obj = (Y*Y+d)^2/2;
    if any(~isfinite([X, Y, tmp, obj]))
        warning('AM_pca overflow at iteration %d; stopped.', i);
        break;
    end
    
    penal = abs(tmp);
    hist_penalty3(i,1) = penal;
    hist_obj3(i,1) = obj;      
    hist_soldist3(i,1) = abs(X); 
    
    gamma = min(1/sqrt(penal), 1e3);
    hist_gamma3(i,1) = gamma;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AM_our %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = X0; Y = Y0; gamma = gamma0;
for i = 1:maxiter
    X = Y*((gamma - d)/(Y*Y + gamma));  
    Y = X*((gamma - d)/(X*X + gamma));     
    tmp = X-Y;     
    
    penal = abs(tmp);
    hist_penalty4(i,1) = penal;
    obj   = (Y*Y+d)^2;
    hist_obj4(i,1) = (Y*Y+d)^2/2;      
    hist_soldist4(i,1) = abs(X);
    
    gamma = max(min((X*Y+d)/2,gamma),(X*Y + d)/nu/2+epsilon);
    hist_gamma4(i,1) = gamma;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid=figure(1); set(fid,'position',[546   347   304   234]);
loglog(1:maxiter,hist_gamma1,'--k',1:maxiter,hist_gamma2,'-.b',1:maxiter,hist_gamma3,'--g',1:maxiter,hist_gamma4(1:end),'-r','LineWidth',1,'MarkerSize',2)
   ylabel('penalty parameter $${\gamma}_k$$','FontSize',12,'Interpreter','latex'); xlabel('number of iterations','FontSize',12);

y1 = min([hist_gamma2;hist_gamma3]); y2 = max([hist_gamma2;hist_gamma3]);
xlim([1,maxiter]); ylim([1e-10,1e17]); 
h = legend('{AM$$_{agd}$$}','{AM$$_{adp}$$}','{AM$$_{pca}$$}','{AM$$_{our}$$}',...
    'Location','southwest','Orientation','vertical','Interpreter','latex');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid=figure(2); set(fid,'position',[546   347   304   234]);
%yyaxis left
loglog(1:maxiter,hist_obj1,'--k',1:maxiter,hist_obj2,'-.b',1:maxiter,hist_obj3,'--g',1:maxiter,hist_obj4(1:end),'-r','LineWidth',1,'MarkerSize',2);
ylabel('$$f(y_k^2)=\frac{1}{2}(y_k^2+1)^2$$','FontSize',12,'Interpreter','latex'); xlabel('number of iterations','FontSize',12);
y1 = min([hist_obj1;hist_obj2]); y2 = max([hist_obj1;hist_obj2]);
xlim([1,maxiter]); ylim([0.47,2.2]);  
h = legend('{AM$$_{agd}$$}','{AM$$_{adp}$$}','{AM$$_{pca}$$}','{AM$$_{our}$$}',...
    'Location','Northwest','Orientation','vertical','Interpreter','latex');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid=figure(3); set(fid,'position',[546   347   304   234]);
%yyaxis left
loglog(1:maxiter,hist_penalty1,'--k',1:maxiter,hist_penalty2,'-.b',1:maxiter,hist_penalty3,'--g',1:maxiter,hist_penalty4(1:end),'-r','LineWidth',1,'MarkerSize',2)
   ylabel('$$\|x_k-y_k\|$$','FontSize',12,'Interpreter','latex'); xlabel('number of iterations','FontSize',12);
y1 = min([hist_penalty1;hist_penalty2;hist_penalty3]); y2 = max([hist_penalty1;hist_penalty2;hist_penalty3]);
ylim([1e-20,1e3]); 
xlim([1,maxiter]);
h = legend('{AM$$_{agd}$$}','{AM$$_{adp}$$}','{AM$$_{pca}$$}','{AM$$_{our}$$}',...
    'Location','southwest','Orientation','vertical','Interpreter','latex');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid=figure(4); set(fid,'position',[546   347   304   234]);
%yyaxis left
loglog(1:maxiter,(hist_soldist1),'--k',1:maxiter,(hist_soldist2),'-.b',1:maxiter,(hist_soldist3),'--g',1:maxiter,(hist_soldist4),'-r','LineWidth',1,'MarkerSize',2)
   ylabel('$$\|x_k-x^\star\|$$','FontSize',12,'Interpreter','latex'); xlabel('number of iterations','FontSize',12);
xlim([1,maxiter]);
ylim([1e-15,1e3]); 
h = legend('{AM$$_{agd}$$}','{AM$$_{adp}$$}','{AM$$_{pca}$$}','{AM$$_{our}$$}',...
    'Location','southwest','Orientation','vertical','Interpreter','latex');
end

