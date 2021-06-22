clc
clear
addpath(genpath(pwd));

Accuracy_lost = 0;

percentage = 0.50;
T_p=strcat('partial_lost_0.5','.mat');

Accuracy_SPL_LSP = SPL_LSP(T_p);
Accuracy_lost = Accuracy_SPL_LSP;






