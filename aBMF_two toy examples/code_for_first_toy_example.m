%%%%%%%% Comparison of four different penalty parameter strategies  %%%%%%%

close all; clear;clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y0 = 100; gamma0 = 1e-5; epsilon = 1e-3; maxiter = 1e4; nu = 0.3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AM_agd %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = y0; gamma = gamma0; 
for k = 1:maxiter
    x = ((gamma - 1)/gamma) * y;    
    y = ((gamma - 1)/gamma) * x; 
    
    tmp = x-y;
    z = y;
    obj = sum(sum(z.*z));
    hist_obj1(k,1) = obj;       
    penal = abs(tmp);
    hist_soldist1(k,1) = abs(x);
    hist_penalty1(k,1) = penal;        
    
    ad = 2*sum(sum(tmp.*x))/sum(sum(tmp.*tmp));
    gamma = max(0, 0 + ad) + epsilon; 
    hist_gamma1(k,1) = gamma;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AM_adp %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = y0; gamma = gamma0;
for k = 1:maxiter
    x = ((gamma - 1)/gamma) * y;
    y = ((gamma - 1)/gamma) * x;
    tmp = x-y;
    
    z = y;
    obj = sum(sum(z.*z));
    hist_obj2(k,1) = obj;       
    
    penal = abs(tmp);
    hist_soldist2(k,1) = abs(x);
    hist_penalty2(k,1) = penal;      
    
    gamma = gamma *((sum(sum(x.*x))+sum(sum(y.*y)))/abs(2*sum(sum(x.*y))));
    hist_gamma2(k,1) = gamma;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AM_pca %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = y0; gamma = gamma0; 
hist_obj3     = nan(maxiter,1);
hist_gamma3   = nan(maxiter,1);
hist_soldist3 = nan(maxiter,1);
hist_penalty3 = nan(maxiter,1);
for k = 1:maxiter
    x = ((gamma - 1)/gamma) * y;
    y = ((gamma - 1)/gamma) * x;
    tmp = x-y;
    
    if any(~isfinite([x, y, tmp, y*y]))
        warning('AM_pca overflow at iteration %d; stopped.', k);
        break;
    end
    
    z = y;
    obj = sum(sum(z.*z));
    hist_obj3(k,1) = obj;       
    
    penal = abs(tmp);
    hist_soldist3(k,1) = abs(x);
    hist_penalty3(k,1) = penal;            

    gamma = 1/sqrt(penal);
    gamma = min(gamma, 1e3);
    hist_gamma3(k,1) = gamma;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AM_our %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = y0; gamma = gamma0; 
for k = 1:maxiter
    x = ((gamma - 1)/gamma) * y;
    y = ((gamma - 1)/gamma) * x;
    tmp = x-y;
    
    z = y;
    obj = sum(sum(z.*z));
    hist_obj4(k,1) = obj;   
        
    penal = abs(tmp);
    hist_soldist4(k,1) = abs(x);
    hist_penalty4(k,1) = penal;  
    
    gamma = max(min(gamma,0.5*1),1/(2*nu)+epsilon);
    hist_gamma4(k,1) = gamma;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% drawing results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid=figure(1); set(fid,'position',[546   347   304   234]);
loglog(1:maxiter,hist_gamma1,'--k',1:maxiter,hist_gamma2,'-.b',1:maxiter,hist_gamma3,'--g',1:maxiter,hist_gamma4(1:end),'-r','LineWidth',1,'MarkerSize',2)
   ylabel('penalty parameter $${\gamma}_k$$','FontSize',12,'Interpreter','latex'); xlabel('number of iterations','FontSize',12);
xlim([1,maxiter]); ylim([1e-10,1e20]); 
h = legend('{AM$$_{agd}$$}','{AM$$_{adp}$$}','{AM$$_{pca}$$}','{AM$$_{our}$$}',...
    'Location','Northwest','Orientation','vertical','Interpreter','latex');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid=figure(3); set(fid,'position',[546   347   304   234]);
loglog(1:maxiter,hist_penalty1,'--k',1:maxiter,hist_penalty2,'-.b',1:maxiter,hist_penalty3,'--g',1:maxiter,hist_penalty4(1:end),'-r','LineWidth',1,'MarkerSize',2)
   ylabel('$$\|x_k-y_k\|$$','FontSize',12,'Interpreter','latex'); xlabel('number of iterations','FontSize',12);
xlim([1,maxiter]); ylim([1e-10,1e25]); 
h = legend('{AM$$_{agd}$$}','{AM$$_{adp}$$}','{AM$$_{pca}$$}','{AM$$_{our}$$}',...
    'Location','southeast','Orientation','vertical','Interpreter','latex');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid=figure(4); set(fid,'position',[546   347   304   234]);
loglog(1:maxiter,(hist_soldist1),'--k',1:maxiter,(hist_soldist2),'-.b',1:maxiter,(hist_soldist3),'--g',1:maxiter,(hist_soldist4),'-r','LineWidth',1,'MarkerSize',2)
   ylabel('$$\|x_k-x^\star\|$$','FontSize',12,'Interpreter','latex'); xlabel('number of iterations','FontSize',12);
xlim([1,maxiter]); ylim([1e-20,1e30]); 
h = legend('{AM$$_{agd}$$}','{AM$$_{adp}$$}','{AM$$_{pca}$$}','{AM$$_{our}$$}',...
    'Location','southeast','Orientation','vertical','Interpreter','latex');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid=figure(2); set(fid,'position',[546   347   304   234]);
loglog(1:maxiter,hist_obj1,'--k',1:maxiter,hist_obj2,'-.b',1:maxiter,hist_obj3,'--g',1:maxiter,hist_obj4(1:end),'-r','LineWidth',1,'MarkerSize',2);
ylabel('$$f(y_k^2)=y_k^2$$','FontSize',12,'Interpreter','latex');xlabel('number of iterations','FontSize',12);
xlim([1,maxiter]); ylim([1e-10,1e180]);  
h = legend('{AM$$_{agd}$$}','{AM$$_{adp}$$}','{AM$$_{pca}$$}','{AM$$_{our}$$}',...
    'Location','Northeast','Orientation','vertical','Interpreter','latex');



