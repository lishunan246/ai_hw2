function [w, num] = svm(X, y)
%SVM Support vector machine.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           num:  number of support vectors
%

% YOUR CODE HERE
[P,N]=size(X);    
x=[ones(1,N); X];
w=ones(P+1,1);

A=-x.*repmat(y,3,1);
b=zeros(N,1);
H=eye(P+1);
f=zeros(P+1,1);
w=quadprog(H,f,A',b);
num=0;

alpha=quadprog((X'*X).*(y'*y),ones(1,N),[],[],y,0,zeros(1,N),[])

end
