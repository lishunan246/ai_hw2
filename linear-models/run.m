% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%% Part1: Preceptron
nRep = 1000; % number of replicates
nTrain = 10; % number of training data

totalIter=0;
totalEtest=0;
nTest=nTrain;
for i = 1:nRep
    [X_all,y_all,w_f]=mkdata(nTrain+nTest);
    
    X=X_all(:,1:nTest);
    y=y_all(:,1:nTest);
    [w_g, iter] = perceptron(X, y);
    % Compute training, testing error
    % Sum up number of iterations
    totalIter=totalIter+iter;
    
    X=X_all(:,nTest+1:nTrain+nTest);
    y=y_all(:,nTest+1:nTrain+nTest);
    [P,N]=size(X);
    a=[ones(1,N); X];
    f=w_g'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error=(f~=y);
    E_test=sum(error);
    totalEtest=totalEtest+E_test;
end
avgIter=totalIter/(1.0*nRep);
E_test=totalEtest/(1.0*nRep*nTest);
fprintf('E_train is %f, E_test is %f.\n', 0, E_test);
fprintf('Average number of iterations is %d.\n', avgIter);
plotdata(X, y, w_f, w_g, 'Preceptron');

%% Part2: Preceptron: Non-linearly separable case
nTrain = 100; % number of training data
[X, y, w_f] = mkdata(nTrain, 'noisy');
%[w_g, iter] = perceptron(X, y);


%% Part3: Linear Regression
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest=nTrain;
totalEtest=0;
totalEtrain=0;
for i = 1:nRep
    [X_all,y_all,w_f]=mkdata(nTrain+nTest);
    X=X_all(:,1:nTrain);
    y=y_all(:,1:nTrain);
    X_test=X_all(:,nTrain+1:nTrain+nTest);
    y_test=y_all(:,nTrain+1:nTrain+nTest);
    
    w_g = linear_regression(X, y);
    % Compute training, testing error
    [P,N]=size(X);
    a=[ones(1,N); X];
    f=w_g'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error=(f~=y);
    totalEtrain=totalEtrain+sum(error);
    
    [P,N]=size(X_test);
    a=[ones(1,N); X_test];
    f=w_g'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error=(f~=y_test);
    totalEtest=totalEtest+sum(error);
end
E_test=totalEtest/(1.0*nRep*nTest);
E_train=totalEtrain/(1.0*nRep*nTrain);
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Linear Regression');

%% Part4: Linear Regression: noisy
nRep = 1000; % number of replicates
nTrain = 100; % number of training data

nTest=nTrain;
totalEtest=0;
totalEtrain=0;
for i = 1:nRep
    [X_all,y_all,w_f]=mkdata(nTrain+nTest,'noisy');
    X=X_all(:,1:nTrain);
    y=y_all(:,1:nTrain);
    X_test=X_all(:,nTrain+1:nTrain+nTest);
    y_test=y_all(:,nTrain+1:nTrain+nTest);
    
    w_g = linear_regression(X, y);
    % Compute training, testing error
    [P,N]=size(X);
    a=[ones(1,N); X];
    f=w_g'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error=(f~=y);
    totalEtrain=totalEtrain+sum(error);
    
    [P,N]=size(X_test);
    a=[ones(1,N); X_test];
    f=w_g'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error=(f~=y_test);
    totalEtest=totalEtest+sum(error);
end
E_test=totalEtest/(1.0*nRep*nTest);
E_train=totalEtrain/(1.0*nRep*nTrain);

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Linear Regression: noisy');

%% Part5: Linear Regression: poly_fit
load('poly_train', 'X', 'y');
load('poly_test', 'X_test', 'y_test');
w = linear_regression(X, y)
% Compute training, testing error


    [~,nTrain]=size(X);
    a=[ones(1,nTrain); X];
    f=w'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error=(f~=y);
    totalEtrain=sum(error);
    
    [~,nTest]=size(X_test);
    a=[ones(1,nTest); X_test];
    f=w'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error=(f~=y_test);
    totalEtest=sum(error);

E_test=totalEtest/(1.0*nTest);
E_train=totalEtrain/(1.0*nTrain);


fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);

% poly_fit with transform
[~,n]=size(X);
X_t = [X; X(1,:).*X(2,:);X(1,:).*X(1,:);X(2,:).*X(2,:) ]; % CHANGE THIS LINE TO DO TRANSFORMATION
X_test_t = [X_test; X_test(1,:).*X_test(2,:);X_test(1,:).*X_test(1,:);X_test(2,:).*X_test(2,:) ]; % CHANGE THIS LINE TO DO TRANSFORMATION
w = linear_regression(X_t, y)
% Compute training, testing error

    [~,nTrain]=size(X_t);
    a=[ones(1,nTrain); X_t];
    f=w'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error=(f~=y);
    totalEtrain=sum(error);
    
    [~,nTest]=size(X_test_t);
    a=[ones(1,nTest); X_test_t];
    f=w'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error=(f~=y_test);
    totalEtest=sum(error);

E_test=totalEtest/(1.0*nTest);
E_train=totalEtrain/(1.0*nTrain);
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);


%% Part6: Logistic Regression
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest=nTrain;
totalEtest=0;
totalEtrain=0;
for i = 1:nRep
    [X_all,y_all,w_f]=mkdata(nTrain+nTest);
    X=X_all(:,1:nTrain);
    
    y=y_all(:,1:nTrain);
    X_test=X_all(:,nTrain+1:nTrain+nTest);
    y_test=y_all(:,nTrain+1:nTrain+nTest);
    
    w_g = logistic(X, y);
    % Compute training, testing error
    
    [P,N]=size(X);
    a=[ones(1,N); X];
    f=w_g'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error=(f~=y);
    totalEtrain=totalEtrain+sum(error);
    
    [P,N]=size(X_test);
    a=[ones(1,N); X_test];
    f=w_g'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error=(f~=y_test);
    totalEtest=totalEtest+sum(error);
end
E_test=totalEtest/(1.0*nRep*nTest);
E_train=totalEtrain/(1.0*nRep*nTrain);

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Logistic Regression');

%% Part7: Logistic Regression: noisy
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest = 10000; % number of training data

totalEtest=0;
totalEtrain=0;
for i = 1:nRep
    [X_all,y_all,w_f]=mkdata(nTrain+nTest,'noisy');
    X=X_all(:,1:nTrain);
    
    y=y_all(:,1:nTrain);
    X_test=X_all(:,nTrain+1:nTrain+nTest);
    y_test=y_all(:,nTrain+1:nTrain+nTest);
    
    w_g = logistic(X, y);
    % Compute training, testing error
    
    [P,N]=size(X);
    a=[ones(1,N); X];
    f=w_g'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error1=(f~=y);
    totalEtrain=totalEtrain+sum(sum(error1));
    
    [P,N]=size(X_test);
    a=[ones(1,N); X_test];
    f=w_g'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error=(f~=y_test);
    totalEtest=totalEtest+sum(sum(error));
end
E_test=totalEtest/(1.0*nRep*nTest);
E_train=totalEtrain/(1.0*nRep*nTrain);

fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Logistic Regression: noisy');

%% Part8: SVM
nRep = 1000; % number of replicates
nTrain = 100; % number of training data

nTest = 1000; % number of training data

totalEtest=0;
totalEtrain=0;
for i = 1:nRep
    [X_all,y_all,w_f]=mkdata(nTrain+nTest);
    X=X_all(:,1:nTrain);
    
    y=y_all(:,1:nTrain);
    X_test=X_all(:,nTrain+1:nTrain+nTest);
    y_test=y_all(:,nTrain+1:nTrain+nTest);
    
    [w_g, num_sc] = svm(X, y);
    % Compute training, testing error
    [P,N]=size(X);
    a=[ones(1,N); X];
    f=w_g'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error1=(f~=y);
    totalEtrain=totalEtrain+sum(sum(error1));
    
    [P,N]=size(X_test);
    a=[ones(1,N); X_test];
    f=w_g'*a;
    f(f>0)=1;
    f(f<=0)=-1;
    error=(f~=y_test);
    totalEtest=totalEtest+sum(sum(error));   
    
    
    % Sum up number of support vectors
end
E_test=totalEtest/(1.0*nRep*nTest);
E_train=totalEtrain/(1.0*nRep*nTrain);
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'SVM');
