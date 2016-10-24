function w = linear_regression(X, y)
%LINEAR_REGRESSION Linear Regression.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE
[P,N]=size(X);    
x=[ones(1,N); X];
w=ones(P+1,1);

for i=1:500
    w=w-0.01*x*(w'*x-y)'/(P+1);
end
end
