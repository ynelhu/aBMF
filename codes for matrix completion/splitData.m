function [trainY, testY] = splitData(Y, p)
%SPLITDATA Splits the input matrix Y randomly into train data and test data.
%   input : p - Determines how much goes into train data trainY (e.g. 0.7
%               or 0.8 usually)

[row, col, y] = find(Y);
[m, n] = size(Y);

mark = (rand(size(row)) > p);

trainY = sparse(row(~mark), col(~mark), y(~mark), m, n);

testY = sparse(row(mark), col(mark), y(mark), m, n);

fprintf('\nnumber of nonzeros in train data %d', nnz(trainY));
fprintf('\nnumber of nonzeros in test  data %d\n', nnz(testY));

end

