function audit_mixed_history(h,nu,epsilon0)
% Formula checks only; this does not assert convergence or model feasibility.
if isstruct(h),h=struct2table(h);end
assert(height(h)>0,'Empty mixed-rule history.');
used=h.gamma_used;next=h.gamma_next;
assert(all(isfinite(used))&&all(used>0)&&all(isfinite(next))&&all(next>0));
if height(h)>1,close_values(used(2:end),next(1:end-1));end
close_values(h.gamma_previous,used);
for k=1:height(h)
    if h.D_zero(k)~=0,close_values(next(k),used(k));
    else
        check=max(h.directional_quotient(k)/(2*nu),0)+epsilon0;
        hat=max(h.loss_gradient_fro(k)/2,0);
        expected=max(min(used(k),hat),check);
        close_values(h.gamma_check(k),check);close_values(h.gamma_hat(k),hat);
        close_values(next(k),expected);
    end
end
end
function close_values(a,b)
assert(all(abs(a(:)-b(:))<=1e-10.*max(1,abs(b(:)))),'Mixed-rule history mismatch.');
end
