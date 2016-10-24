function w = logistic(X, y)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE

[P,N]=size(X);    
x=[ones(1,N); X];
w=zeros(P+1,1);
y(y==-1)=0;


for i=1:2000
    tmp=w'*x;
    g=1.0./(1+exp(-tmp));
    w=w-0.01*x*(g-y)'/(P+1);
end

end
