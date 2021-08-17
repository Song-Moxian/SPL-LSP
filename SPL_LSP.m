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
% 
% [pl_target] = partial_label_assignment(labeled_data, labeled_target, unlabeled_data, k);
% 
% [pl_model, sl_model, mu_pl, mu_sl] = EUPAL_train(labeled_data, labeled_target, unlabeled_data, pl_target, alpha, T);
% 
% [accuracy, pre_label, pre_value] = EUPAL_predict(test_data, test_target, pl_model, sl_model, mu_pl, mu_sl);
%%%%%%%%%%%%%%%%%%IPAL
% k = 10;                  %set the number of nearest neighbors
% alpha = 0.95;            %set the balancing coefficient
% 
% model = IPAL_train1(labeled_data,labeled_target',k,alpha);                     %disambiguation phase 
% [accuracy,predictLabel] = IPAL_predict1(model,labeled_data,train_clear_target',k);      %testing phase
% fprintf('classification accuracy: %f\n',accuracy);
%%%%%%%%%%%%%%%%%%%%%%%%LSP
k = 5;
alpha = 0.4;
T=50;

[pl_target] = partial_label_assignment_test(labeled_data, labeled_target, unlabeled_data, k);
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

%%%%%%%%%%%%%%%%%IPAL

% k = 10;                  %set the number of nearest neighbors
% alpha = 0.95;            %set the balancing coefficient
% 
% model = IPAL_train1(train_data_all,train_taget_all',k,alpha);                     %disambiguation phase 
% [accuracy,predictLabel] = IPAL_predict1(model,testData,testTarget',k);      %testing phase
% fprintf('classification accuracy: %f\n',accuracy);
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%aggd on pl
% ker  = 'rbf'; %Type of kernel function
% k = 10; %Number of neighbors
% lambda = 1;
% mu = 1;
% gama = 0.05;
% Maxiter = 10;
% par = 1*mean(pdist(labeled_data)); %Parameters of kernel function
% %training
% test_outputs = PL_AGGD(labeled_data,labeled_target,testData,k,ker,par,Maxiter,lambda,mu,gama);
% accuracy = CalAccuracy(test_outputs, testTarget);
% fprintf('The accuracy of PL-AGGD is: %f \n',accuracy);
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%IPALon pl
% 
% k = 10;                  %set the number of nearest neighbors
% alpha = 0.95;            %set the balancing coefficient
% 
% model = IPAL_train1(labeled_data,labeled_target',k,alpha);                     %disambiguation phase 
% [accuracy,predictLabel] = IPAL_predict1(model,testData,testTarget',k);      %testing phase
% fprintf('classification accuracy: %f\n',accuracy);
% 
% 
% 
% 
% 
% 
% 
% 
% 
% labeled_data = labeled_data;
% labeled_target = predictLabel';
% unlabeled_data = unlabeled_data;
% 
% 
% gimma = 2;
% L = size(labeled_data,1);
% U = size(unlabeled_data,1);
% class_num = size(labeled_target, 2);
% 
% k=min(k,L);
% 
% unlabel_space = zeros(U, class_num);
% partial_target = zeros(U, class_num);
% 
% kdtree = KDTreeSearcher(labeled_data);
% neighbor = knnsearch(kdtree,unlabeled_data,'k',k);%u x k
% 
% w = zeros(U, k);
% 
% for i = 1:U
%     for j = 1:k
%         w(i,j) = exp(-(norm(unlabeled_data(i,:)-labeled_data(neighbor(i,j),:),2)^2)/gimma);
%     end
%     unlabel_space(i,:) = w(i,:) * labeled_target(neighbor(i,:),:);
%     sum_label = sum(unlabel_space(i,:));
%     if sum_label == 0
%         partial_target(i,:) = ones(1, class_num);
%     else
%         unlabel_space(i,:) = unlabel_space(i,:)/sum_label;
%         partial_target(i,unlabel_space(i,:)>=(1/class_num)) = 1;%
%     end
% end
% 
% ker  = 'rbf'; %Type of kernel function
% k = 10; %Number of neighbors
% lambda = 1;
% mu = 1;
% gama = 0.05;
% Maxiter = 10;
% par = 1*mean(pdist(train_data)); %Parameters of kernel function
% %training
% test_outputs = PL_AGGD(train_data,train_p_target,test_data,k,ker,par,Maxiter,lambda,mu,gama);
% accuracy = CalAccuracy(test_outputs, test_target);
% fprintf('The accuracy of PL-AGGD is: %f \n',accuracy);
% 
% 
% 
% 
% 
% 
% 
% 
% clear;
% load('sample_data.mat');
% labeled_data = data(tr_label, :);
% labeled_target = target(tr_label, :);
% unlabeled_data = data(tr_unlabel, :);
% test_data = data(test, :);
% test_target = target(test,: );
% 
% k = 5;
% alpha = 0.4;
% T=50;
% 
% 
% gimma = 2;
% L = size(labeled_data,1);
% U = size(unlabeled_data,1);
% class_num = size(labeled_target, 2);
% 
% k=min(k,L);
% 
% unlabel_space = zeros(U, class_num);
% partial_target = zeros(U, class_num);
% 
% kdtree = KDTreeSearcher(labeled_data);
% neighbor = knnsearch(kdtree,unlabeled_data,'k',k);%u x k
% 
% w = zeros(U, k);
% 
% for i = 1:U
%     for j = 1:k
%         w(i,j) = exp(-(norm(unlabeled_data(i,:)-labeled_data(neighbor(i,j),:),2)^2)/gimma);
%     end
%     unlabel_space(i,:) = w(i,:) * labeled_target(neighbor(i,:),:);
%     sum_label = sum(unlabel_space(i,:));
% %     if sum_label == 0
% %         partial_target(i,:) = ones(1, class_num);
% %     else
% %         unlabel_space(i,:) = unlabel_space(i,:)/sum_label;
% %         partial_target(i,unlabel_space(i,:)>=(1/class_num)) = 1;%
% %     end
% end
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% [pl_target] = partial_label_assignment(labeled_data, predictLabel', unlabeled_data, k);
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% load 'lost_partial_4_0.5.mat';
% data = zscore(data);
% load('sample data.mat');
% labeled_data = data(tr_label, :);
% labeled_target = target(tr_label, :);
% unlabeled_data = data(tr_unlabel, :);
% test_data = data(test, :);
% test_target = target(test,: );
% 
% % Set parameters for the EUPAL algorithm
% k = 5;
% alpha = 0.4;
% T=50;
% 
% load('sample data.mat'); % Loading the file containing the necessary inputs for calling the IPAL function
% 
% k = 10;                  %set the number of nearest neighbors
% alpha = 0.95;            %set the balancing coefficient
% 
% model = IPAL_train1(labeled_data,labeled_target,k,alpha);                     %disambiguation phase 
% [accuracy,predictLael] = IPAL_predict1(model,labeled_data,labeled_truelabel,k);      %testing phase
% fprintf('classification accuracy: %f\n',accuracy);
% 
% 
% 
% [pl_target] = partial_label_assignment(labeled_data, labeled_target, unlabeled_data, k);
% 
% 
% 
% 
% [pl_model, sl_model, mu_pl, mu_sl] = EUPAL_train(labeled_data, labeled_target, unlabeled_data, pl_target, alpha, T);
% 
% [accuracy, pre_label, pre_value] = EUPAL_predict(test_data, test_target, pl_model, sl_model, mu_pl, mu_sl);
% 
% 
% 
% % Call the main functions
% [accuracy, pre_label, pre_value] = EUPAL(labeled_data, labeled_target, unlabeled_data, test_data, test_target, k, alpha, T);
% disp(accuracy);