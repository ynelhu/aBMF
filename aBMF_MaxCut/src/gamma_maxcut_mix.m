function [next,info] = gamma_maxcut_mix(graph,X,Y,diagonal,previous,nu,epsilon0,LX,LY)
% G=-L_work/4+Diag(lambda+rho*(diag(X*Y')-1)); sigma_h=0.
D=X-Y;dn=norm(D,'fro');zero=~any(D(:));
norm_sq=graph.gradient_fro^2+sum(diagonal.^2)-0.5*sum(graph.L_work_diagonal.*diagonal);
gradient_fro=sqrt(max(norm_sq,0));
if zero
    rayleigh=NaN;
else
    U=D/dn;
    if nargin>=9&&dn/max([1,norm(X,'fro'),norm(Y,'fro')])>sqrt(eps)
        LU=(LX-LY)/dn;
    else
        LU=graph.L_work*U;
    end
    rayleigh=-0.25*sum(sum(U.*LU))+sum(diagonal.*sum(U.^2,2));
end
[next,info]=mixed_gamma_update(previous,rayleigh,gradient_fro,zero,nu,epsilon0,0);
info.D_norm_sq=dn^2;
end
