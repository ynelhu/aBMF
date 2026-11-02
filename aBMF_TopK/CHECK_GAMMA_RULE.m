function report = CHECK_GAMMA_RULE
% Dense-reference tests for the signed directional and norm-based candidates.
state=rng;state_guard=onCleanup(@()rng(state)); %#ok<NASGU>
rng(20261101,'twister');count=0;max_error=0;
settings=struct('nu',0.9,'epsilon0',1e-6,'sigma_h',0);
for j=1:8
    n=9;r=3;A=randn(n);A=0.5*(A+A');
    if mod(j,2)==0,A=A*A';end
    X=randn(n,r);Y=randn(n,r);previous=0.2+j;
    if j==7,Y=X+1e-10*randn(n,r);end
    if j==8,settings.sigma_h=0.4;end
    data=struct('apply',@(V)A*V,'fro_sq',norm(A,'fro')^2);
    G=X*Y'-A;D=X-Y;
    quotient=trace(D'*G*D)/norm(D,'fro')^2;
    directional=max(quotient/(2*settings.nu)-settings.sigma_h/4,0)+settings.epsilon0;
    norm_based=max(norm(G,'fro')/2-settings.sigma_h/4,0);
    expected=max(min(previous,norm_based),directional);
    cache=struct('AX',A*X,'AY',A*Y,'loss_fro',norm(G,'fro'));
    for use_cache=0:1
        if use_cache,[actual,info]=gamma_mix_topk(previous,X,Y,data,settings,cache);
        else,[actual,info]=gamma_mix_topk(previous,X,Y,data,settings);end
        errors=[abs(actual-expected)/max(1,abs(expected)), ...
            abs(info.directional_quotient-quotient)/max(1,abs(quotient)), ...
            abs(info.gamma_hat-norm_based)/max(1,abs(norm_based)), ...
            abs(info.gamma_check-directional)/max(1,abs(directional))];
        assert(all(errors<1e-9),'topk:selfcheck','Mixed rule differs from dense reference.');
        max_error=max([max_error,errors]);count=count+1;
        if j==7,assert(info.operator_fallback,'Tiny D must use the stable operator evaluation.');end
    end
end
settings.sigma_h=0;
% Exact D=0 retains the previous value, even below epsilon0.
A=eye(4);X=ones(4,1);data=struct('apply',@(V)A*V,'fro_sq',norm(A,'fro')^2);
[value,info]=gamma_mix_topk(1e-8,X,X,data,settings);
assert(value==1e-8&&info.D_zero);count=count+1;
% Negative directional value: positive part is taken BEFORE adding epsilon0.
A=10*eye(4);X=[2;0;0;0];Y=[1;0;0;0];data.apply=@(V)A*V;data.fro_sq=norm(A,'fro')^2;
[~,info]=gamma_mix_topk(1,X,Y,data,settings);
assert(info.directional_quotient<0&&info.gamma_check==settings.epsilon0);count=count+1;
% A large directional safeguard may increase gamma; it must not be clipped.
A=-10*eye(4);data.apply=@(V)A*V;data.fro_sq=norm(A,'fro')^2;
[value,info]=gamma_mix_topk(1e-8,X,Y,data,settings);
assert(info.gamma_check>1e-8&&value==info.gamma_check);count=count+1;
% A permitted decrease uses the norm candidate, while retaining the safeguard.
A=diag([0.2,0.2,0.3,0.4]);Y=[0.5;0;0;0];X=[1;0;0;0];
data.apply=@(V)A*V;data.fro_sq=norm(A,'fro')^2;
[value,info]=gamma_mix_topk(10,X,Y,data,settings);
assert(value<10&&value==info.gamma_hat&&value>=info.gamma_check);count=count+1;
% With normal nonzero D and a complete cache, the helper needs no A product.
cache=struct('AX',A*X,'AY',A*Y,'loss_fro',norm(X*Y'-A,'fro'));
data.apply=@(V)error('topk:unexpectedOperator','Unexpected extra operator product.');
gamma_mix_topk(10,X,Y,data,settings,cache);count=count+1;
% A small end-to-end test exercises MATLAB CSV, solver, and history interfaces.
folder=tempname;mkdir(folder);folder_guard=onCleanup(@()rmdir(folder,'s')); %#ok<NASGU>
for j=1:2
    modes={'explicit','matrix_free'};types={'PSD','Indefinite'};
    output=fullfile(folder,sprintf('test_%d',j));
    run_topk_benchmark('fixed_n',32,'rank_list',3,'fixed_rank',3,'n_list',32, ...
        'matrix_types',types(j),'active_sweeps',{'fixed_n_rank_sweep'}, ...
        'num_trials',1,'max_iter',6,'tol',1e-12,'data_mode',modes{j}, ...
        'nu',settings.nu,'epsilon0',settings.epsilon0, ...
        'include_projection',false,'save_histories',true,'make_plots',false,'output_dir',output);
    AUDIT_GAMMA_HISTORY(output,settings,1,4);count=count+1;
end
report=struct('passed',true,'cases_checked',count,'max_dense_reference_relative_error',max_error, ...
    'dense_gradient_reference',true,'cached_and_uncached',true,'D_zero',true, ...
    'near_zero_D_not_frozen',true,'increase_and_decrease',true, ...
    'explicit_and_matrix_free_solver',true,'matlab_version',version);
fprintf('GAMMA_SELF_CHECK: passed (%d checks).\n',count);
end
