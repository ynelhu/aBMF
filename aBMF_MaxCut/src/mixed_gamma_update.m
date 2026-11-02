function [next,info] = mixed_gamma_update(previous,rayleigh,gradient_fro,D_zero,nu,epsilon0,sigma_h)
% Latest directional/norm mixed rule. rayleigh=tr(D'*G*D)/||D||_F^2.
if nargin<7,sigma_h=0;end
assert(isscalar(previous)&&isfinite(previous)&&previous>0,'Invalid previous gamma.');
assert(isscalar(nu)&&isfinite(nu)&&nu>0&&nu<1,'Require 0<nu<1.');
assert(isscalar(epsilon0)&&isfinite(epsilon0)&&epsilon0>0,'epsilon0 must be positive.');
assert(isfinite(gradient_fro)&&gradient_fro>=0&&isfinite(sigma_h)&&sigma_h>=0,'Invalid norm or sigma_h.');
hat=max(gradient_fro/2-sigma_h/4,0);
if D_zero
    check=NaN;next=previous;rayleigh=NaN;
else
    assert(isfinite(rayleigh),'Nonfinite directional quotient.');
    check=max(rayleigh/(2*nu)-sigma_h/4,0)+epsilon0;
    next=max(min(previous,hat),check);
end
assert(isfinite(next)&&next>0,'Invalid mixed penalty.');
info=struct('gamma_used',previous,'gamma_previous',previous,'gamma_next',next, ...
    'gamma_hat',hat,'gamma_check',check,'directional_quotient',rayleigh, ...
    'loss_gradient_fro',gradient_fro,'D_zero',double(D_zero));
end
