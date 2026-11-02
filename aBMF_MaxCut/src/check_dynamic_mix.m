function check_dynamic_mix
% Explicit small-matrix reference; independent of formal experiment outcomes.
state=rng;guard=onCleanup(@()rng(state)); %#ok<NASGU>
rng(20261101,'twister');nu=0.9;epsilon0=1e-6;
for k=1:20
    n=9;r=3;A=randn(n);A=(A+A')/2;X=randn(n,r);Y=randn(n,r);
    if k==19,Y=X+1e-10*randn(n,r);end
    if k==20,Y=X;end
    D=X-Y;zero=~any(D(:));G=randn(n);G=(G+G')/2;
    if zero,q=NaN;expected=2;
    else,q=trace(D'*G*D)/norm(D,'fro')^2;expected=max(min(2,norm(G,'fro')/2),max(q/(2*nu),0)+epsilon0);end
    value=mixed_gamma_update(2,q,norm(G,'fro'),zero,nu,epsilon0,0);
    assert(abs(value-expected)<=1e-10*max(1,abs(expected)));
    if exist('gamma_maxcut_mix','file')==2
        L=sparse(A);diagonal=randn(n,1);gradient=-L/4+spdiags(diagonal,0,n,n);
        graph=struct('L_work',L,'L_work_diagonal',diag(L),'gradient_fro',norm(L,'fro')/4);
        [actual,info]=gamma_maxcut_mix(graph,X,Y,diagonal,2,nu,epsilon0,L*X,L*Y);
        if zero,q=NaN;else,q=trace(D'*gradient*D)/norm(D,'fro')^2;end
        ref=mixed_gamma_update(2,q,norm(gradient,'fro'),zero,nu,epsilon0,0);
        assert(abs(actual-ref)<=1e-9*max(1,abs(ref)));
        if ~zero,assert(abs(info.directional_quotient-q)<=1e-8*max(1,abs(q)));end
    end
end
assert(mixed_gamma_update(1e-9,NaN,100,true,nu,epsilon0,0)==1e-9);
[value,info]=mixed_gamma_update(1,-5,10,false,nu,epsilon0,0);assert(info.gamma_check==epsilon0);
[value,info]=mixed_gamma_update(1e-9,10,12,false,nu,epsilon0,0);assert(value==info.gamma_check&&value>1e-9);
[value,info]=mixed_gamma_update(10,0.1,1,false,nu,epsilon0,0);assert(value<10&&value==info.gamma_hat);
fprintf('Mixed gamma dense-reference checks passed.\n');
end
