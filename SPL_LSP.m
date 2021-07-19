function Accuracy_SPL_LSP = SPL_LSP(data_path)
% Load the file containing the necessary inputs for calling the EUPAL function
%load 'lost_partial_9_0.5.mat';
load(data_path);
labeled_data = partialData;
%labeled_data = zscore(labeled_data);
labeled_target = partialTarget;
unlabeled_data = unlabeledData;
%unlabeled_data = zscore(unlabeled_data);
test_data = testData;
test_target = testTarget;

%%%%%%%%%%%%%%%%%%%%%%%%LSP
k = 5;
alpha = 0.4;
T=50;

[pl_target] = partial_label_assignment(labeled_data, labeled_target, unlabeled_data, k);
%%%%%%%%%%%%%%%%%%%%%%%data_all
train_data_all = [labeled_data;unlabeled_data];
train_taget_all = [labeled_target;pl_target];

%%%%%%%%%%%%%%%%AGGD

ker  = 'rbf'; %Type of kernel function
k = 10; %Number of neighbors
lambda = 1;
mu = 1;
gama = 0.05;
Maxiter = 10;
par = 1*mean(pdist(train_data_all)); %Parameters of kernel function
%training
test_outputs = premodel(train_data_all,train_taget_all,testData,k,ker,par,Maxiter,lambda,mu,gama);
accuracy = CalAccuracy(test_outputs, testTarget);
fprintf('The accuracy of SPL_LSP is: %f \n',accuracy);

Accuracy_SPL_LSP = accuracy;

