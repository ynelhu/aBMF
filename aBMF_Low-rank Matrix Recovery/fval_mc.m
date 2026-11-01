function [f,c] = fval_mc(Y, QlY, QrY, labs, eta)
c = (sum(bsxfun(@times, QlY, QrY),2)) - labs;
f = (c'*c)/2 + eta*sum(sum(Y.*Y));
end