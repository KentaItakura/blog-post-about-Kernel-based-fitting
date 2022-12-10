clear;clc;close all
%% パラメータ設定
% ガウシアンカーネル中のベータの値
beta = 1;
% 正則化のパラメータ
L2regu = 0;
%% サンプルデータの作成と計算の準備
rng('default') % For reproducibility
x_observed = linspace(0,10,21)';
y_observed1 = x_observed.*sin(x_observed);
y_observed2 = y_observed1 + 0.5*randn(size(x_observed));
numSample = numel(x_observed);
K = zeros(numSample, numSample);
%% グラム行列の計算
% わかりやすくfor文を2つ用いて実装します
for i =1:numSample
    for j = 1:numSample

        diff = (x_observed(i)-x_observed(j))^2;
        K(i,j) = exp(-beta*diff);
    end
end
%% 重みwの計算
w = inv((K+L2regu*eye(numSample)))*y_observed2;
%% 結果の可視化
% サンプルデータのプロット
figure;scatter(x_observed,y_observed2);hold on
% フィッティングした結果のプロット
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





