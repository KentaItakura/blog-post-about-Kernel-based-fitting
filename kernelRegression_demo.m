% 正則化項のパラメータを変えながらフィッティングの様子を可視化するコード

clear;clc;close all

beta = 1;
%% サンプルデータの作成と計算の準備
rng('default') % For reproducibility
x_observed = linspace(0,10,21)';
y_observed1 = x_observed.*sin(x_observed);
y_observed2 = y_observed1 + 0.5*randn(size(x_observed));
numSample = numel(x_observed);
K = zeros(numSample, numSample);
% サンプルデータのプロット
figure;
idx = 1;
filename='output.gif';
DelayTime=0.005;

for L2regu = 1.001:-0.01:0.001

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

% フィッティングした結果のプロット
numSample_test = 100;
x_test = linspace(0,10,numSample_test)';
kOut = zeros(numSample_test,1);
for i =1:numSample_test
    x_i = x_test(i);
    diff_test = (x_observed-x_i).^2;
    kOut(i) = w'*exp(-diff_test.*beta);
end

scatter(x_observed,y_observed2);hold on
plot(x_test,kOut)
title(sprintf('L2正則化の係数=%d',L2regu));hold off;drawnow;pause(0.02)
figData = getframe(gcf);
im = figData.cdata;

[A,map] = rgb2ind(im,256);
if idx == 1
    imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',DelayTime)
else
    imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',DelayTime);
end
% idxの更新
idx=idx+1;

end