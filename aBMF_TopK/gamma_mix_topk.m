function [gamma_next,info] = gamma_mix_topk(previous,X,Y,data,settings,cache)
%GAMMA_MIX_TOPK Exact mixed penalty rule for f(Z)=0.5*||Z-A||_F^2, symmetric A.
% G=X*Y'-A. Frobenius norm is evaluated from r-by-r Grams, not a dense G.
% tr(D'*G*D)/||D||_F^2 is evaluated with U=D/||D||_F for numerical stability.
% cache is optional; AX, AY and loss_fro can be reused from the solver.
if nargin<6,cache=struct();end
assert(isscalar(previous)&&isfinite(previous)&&previous>0,'topk:gamma','previous must be positive.');
assert(isscalar(settings.nu)&&isfinite(settings.nu)&&settings.nu>0&&settings.nu<1,'topk:nu','Require 0<nu<1.');
assert(isscalar(settings.epsilon0)&&isfinite(settings.epsilon0)&&settings.epsilon0>0,'topk:epsilon','epsilon0 must be positive.');
assert(isscalar(settings.sigma_h)&&isfinite(settings.sigma_h)&&settings.sigma_h>=0,'topk:sigma','sigma_h must be nonnegative.');
assert(isequal(size(X),size(Y))&&all(isfinite(X(:)))&&all(isfinite(Y(:))),'topk:factors','Invalid factors.');
D=X-Y;
scale=max(abs(D(:)));
info=struct('D_zero',false,'D_norm',0,'D_norm_sq',0,'directional_numerator',NaN, ...
    'directional_quotient',NaN,'loss_gradient_fro',NaN,'gamma_hat',NaN,'gamma_check',NaN, ...
    'gamma_previous',previous,'gamma_next',previous,'operator_fallback',false);
if scale==0
    % Exactly D=0: retain previous, irrespective of a norm-only candidate.
    info.D_zero=true;gamma_next=previous;return;
end
assert(isfinite(scale),'topk:difference','Nonfinite X-Y.');
U=D/scale;scaled_norm=norm(U,'fro');U=U/scaled_norm;
dnorm=scale*scaled_norm;
info.D_norm=dnorm;info.D_norm_sq=dnorm*dnorm;
if isfield(cache,'loss_fro')
    gradient_fro=cache.loss_fro;
else
    if isfield(cache,'AY'),AY=cache.AY;else,AY=data.apply(Y);end
    GX=X'*X;GY=Y'*Y;
    residual_sq=data.fro_sq+sum(sum(GX.*GY))-2*sum(sum(X.*AY));
    gradient_fro=sqrt(max(residual_sq,0));
end
relative_d=dnorm/max([1,norm(X,'fro'),norm(Y,'fro')]);
if isfield(cache,'AX')&&isfield(cache,'AY')&&relative_d>sqrt(eps)
    % Reuse A*X and A*Y; no additional operator multiplication normally.
    AU=(cache.AX-cache.AY)/dnorm;
else
    % For tiny nonzero D, avoid cancellation in A*X-A*Y. Do NOT freeze gamma.
    AU=data.apply(U);info.operator_fallback=true;
end
P=X'*U;Q=Y'*U;
rayleigh=sum(sum(P.*Q))-sum(sum(U.*AU));
assert(isfinite(gradient_fro)&&gradient_fro>=0&&isfinite(rayleigh),'topk:gamma','Nonfinite candidate inputs.');
directional=max(rayleigh/(2*settings.nu)-settings.sigma_h/4,0)+settings.epsilon0;
norm_based=max(0.5*gradient_fro-settings.sigma_h/4,0);
gamma_next=max(min(previous,norm_based),directional);
assert(isfinite(gamma_next)&&gamma_next>0,'topk:gamma','Nonfinite or nonpositive penalty.');
info.directional_quotient=rayleigh;
info.directional_numerator=rayleigh*info.D_norm_sq;
info.loss_gradient_fro=gradient_fro;
info.gamma_hat=norm_based;info.gamma_check=directional;info.gamma_next=gamma_next;
end
