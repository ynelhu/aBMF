function [checks,raw] = AUDIT_GAMMA_HISTORY(output,settings,expected_trials,expected_records)
% Independently reconstruct every recorded candidate and mixed update.
raw=readtable(fullfile(output,'all_trial_results.csv'));
raw.sweep=string(raw.sweep);raw.matrix_type=string(raw.matrix_type);raw.method=string(raw.method);
assert(height(raw)==expected_records,'topk:records','Unexpected method record count.');
assert(height(unique(raw(:,{'sweep','matrix_type','n','rank','method','trial'})))==height(raw),'topk:duplicate','Duplicate records.');
our=raw(strcmp(raw.method,'aBMF-our'),:);
assert(height(our)==expected_trials,'topk:records','Missing our trials.');
checks=repmat(struct('n',0,'rank',0,'trial',0,'iterations',0,'gamma_first_used',0, ...
    'gamma_last_used',0,'gamma_last_next',0,'increases',0,'decreases',0,'D_zero_rows',0, ...
    'operator_fallback_rows',0,'max_mixed_rule_error',0,'passed',true),height(our),1);
for j=1:height(our)
    t=our(j,:);folder=sprintf('%s_n%d_r%d_%s',char(t.sweep),t.n,t.rank,lower(char(t.matrix_type)));
    h=readtable(fullfile(output,'histories',folder,sprintf('trial_%02d_aBMF_our.csv',t.trial)));
    count=height(h);assert(count>0&&isequal(h.iteration,(1:count)'),'topk:iteration','Invalid iteration sequence.');
    assert(all(isfinite(h.gamma))&&all(h.gamma>0)&&all(isfinite(h.gamma_next))&&all(h.gamma_next>0),'topk:gamma','Nonfinite gamma.');
    assert_close(h.gamma_previous,h.gamma);
    assert_close(h.D_norm_sq,h.disagreement.^2);
    assert_close(h.full_objective,h.data_objective+0.5*h.gamma.*h.disagreement.^2);
    if count>1,assert_close(h.gamma(2:end),h.gamma_next(1:end-1));end
    max_error=0;
    for k=1:count
        if h.D_zero(k)~=0
            assert_close(h.gamma_next(k),h.gamma(k));
            assert(h.disagreement(k)==0,'topk:zero','D=0 flag inconsistent.');
        else
            assert_close(h.loss_gradient_fro(k)^2,2*h.data_objective(k));
            assert_close(h.directional_numerator(k),h.directional_quotient(k)*h.D_norm_sq(k));
            directional=max(h.directional_quotient(k)/(2*settings.nu)-settings.sigma_h/4,0)+settings.epsilon0;
            norm_based=max(h.loss_gradient_fro(k)/2-settings.sigma_h/4,0);
            mixed=max(min(h.gamma(k),norm_based),directional);
            assert_close(h.gamma_check(k),directional);assert_close(h.gamma_hat(k),norm_based);
            assert_close(h.gamma_next(k),mixed);
            assert(h.gamma_next(k)>=directional-1e-10*max(1,directional),'topk:bound','Directional safeguard violated.');
            assert(h.gamma_next(k)<=max(h.gamma(k),directional)+1e-10*max(1,mixed),'topk:bound','Upper update bound violated.');
            max_error=max(max_error,abs(h.gamma_next(k)-mixed)/max(1,abs(mixed)));
        end
    end
    assert_close(t.final_gamma,h.gamma(end));assert_close(t.final_gamma_next,h.gamma_next(end));
    assert(t.iterations==count,'topk:iteration','Final iteration mismatch.');
    delta=h.gamma_next-h.gamma;
    checks(j)=struct('n',t.n,'rank',t.rank,'trial',t.trial,'iterations',count, ...
        'gamma_first_used',h.gamma(1),'gamma_last_used',h.gamma(end),'gamma_last_next',h.gamma_next(end), ...
        'increases',sum(delta>1e-10.*max(1,abs(h.gamma))), ...
        'decreases',sum(delta<-1e-10.*max(1,abs(h.gamma))), ...
        'D_zero_rows',sum(h.D_zero~=0),'operator_fallback_rows',sum(h.operator_fallback~=0), ...
        'max_mixed_rule_error',max_error,'passed',true);
end
numeric_names={'iterations','time','data_objective','rel_objective','subspace_error','rel_disagreement'};
for j=1:numel(numeric_names)
    assert(all(isfinite(raw.(numeric_names{j}))),'topk:nonfinite','Nonfinite result metric.');
end
end

function assert_close(a,b)
assert(all(isfinite(a(:)))&&all(isfinite(b(:)))&&all(abs(a(:)-b(:))<=1e-10*max(1,abs(b(:)))), ...
    'topk:gammaAudit','Saved mixed-rule values are inconsistent.');
end
