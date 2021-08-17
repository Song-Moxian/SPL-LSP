function [partial_target] = partial_label_assignment_test(labeled_data, labeled_target, unlabeled_data, k)

% Syntax
%
%       [partial_target] = partial_label_assignment(labeled_data, labeled_target, unlabeled_data, k)
%
% Description
%
%       EUPAL takes,
%           labeled_data        - A LxD array, the ith instance of labeled instance is stored in labeled_data(i,:)
%           labeled_target      - A LxQ array, if the jth class label is the ground-truth labels for the ith labeled instance, then labeled_target(i,j) equals +1, otherwise labeled_target(i,j) equals 0
%           unlabeled_data      - A UxD array, the ith instance of unlabeled instance is stored in unlabeled_data(i,:)
%           k                   - The number of nearest neighbors used for kNN aggregation (defalut 5)
%      
%       and returns,
%           partial_target           - A UxQ array, if the jth class label is the candidate labels for the ith unlabeled instance, then partial_target(i,j) equals +1, otherwise pl_target(i,j) equals 0

if nargin<4
    k=5;
end
if nargin<3
    error('Not enough input parameters, please check again.');
end
if size(labeled_data,1)~=size(labeled_target,1)
    error('Length of label vector does match the number of instances');
end

% clc;clear;
% labeled_data = partialData;
% unlabeled_data = unlabeledData;
% labeled_target = partialTarget;
% k=5;

gimma = 2;
L = size(labeled_data,1);
U = size(unlabeled_data,1);
class_num = size(labeled_target, 2);

k=min(k,L);

unlabel_space = zeros(U, class_num);
partial_target = zeros(U, class_num);

kdtree = KDTreeSearcher(labeled_data);
neighbor = knnsearch(kdtree,unlabeled_data,'k',k);%u x k

w = zeros(U, k);


Wpu = sparse(U, L);
for i = 1:U
    neighborIns = labeled_data(neighbor(i,:),:)';
    wij = lsqnonneg(neighborIns,unlabeled_data(i,:)');
    Wpu(i,neighbor(i,:)) = wij';
end
sumWpu = sum(Wpu, 2);
sumWpu(sumWpu == 0) = 1;
H = Wpu ./ repmat(sumWpu, 1, L);
H = sparse(H);


for i = 1:U
    for j = 1:k
        w(i,j) = H(i,neighbor(i,j));
        %w(i,j) = exp(-(norm(unlabeled_data(i,:)-labeled_data(neighbor(i,j),:),2)^2)/gimma);
    end
    unlabel_space(i,:) = w(i,:) * labeled_target(neighbor(i,:),:);
    sum_label = sum(unlabel_space(i,:));
    if sum_label == 0
        partial_target(i,:) = ones(1, class_num);
    else
        unlabel_space(i,:) = unlabel_space(i,:)/sum_label;
        partial_target(i,unlabel_space(i,:)>=(1/class_num)) = 1;%
    end
end
end
