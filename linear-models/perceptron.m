function [w, iter] = perceptron(X, y)
%PERCEPTRON Perceptron Learning Algorithm.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           iter: number of iterations
%

% YOUR CODE HERE
    P=0;
    iter=0;
    [P,N]=size(X);
    alpha=0.1;
    a=[ones(1,N); X];
    w=ones(P+1,1);
    c = true;
    while c
        f=w'*a; 
        f(f>0)=1;
        f(f<=0)=-1;
        d=y-f;
        if isequal(d,zeros(1,N))
            c=false;
        else
            w=w+alpha*(a*d');
            iter=iter+1;
        end 
    end
end
