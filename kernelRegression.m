clear;clc;close all

beta = 1;
L2regu = 0;

rng('default') % For reproducibility
x_observed = linspace(0,10,21)';
y_observed1 = x_observed.*sin(x_observed);
y_observed2 = y_observed1 + 0.5*randn(size(x_observed));
numSample = numel(x_observed);
K = zeros(numSample, numSample);

for i =1:numSample
    for j = 1:numSample
%         diff = (y_observed2(i)-y_observed2(j))^2;
        diff = (x_observed(i)-x_observed(j))^2;
        K(i,j) = exp(-beta*diff);
    end
end

w = inv((K+L2regu*eye(numSample)))*y_observed2;

figure;scatter(x_observed,y_observed2);hold on

numSample_test = 100;
x_test = linspace(0,10,numSample_test)';
kOut = zeros(numSample_test,1);
for i =1:numSample_test

    x_i = x_test(i);
    diff_test = (x_observed-x_i).^2;
    kOut(i) = w'*exp(-diff_test.*beta);
    
end

plot(x_test,kOut)
title(sprintf('L2正則化の係数=%d',L2regu));hold off





