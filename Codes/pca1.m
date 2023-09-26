clc;
clear;
close all;

% Generate Data
W = randn(1000, 2);
A = rand(2, 2);
X = W * A;

% Covariance Matrix
C = cov(X);

% Find PC1
[V, E] = eig(C);
e = diag(E);
[emax, emax_ind] = max(e);
u = V(:, emax_ind);

% Transform Data
z = X * u;

% Decode Data
Y = z * u';

% Plot Results
figure;
scatter(X(:, 1), X(:, 2));
hold on;
scatter(Y(:, 1), Y(:, 2));
