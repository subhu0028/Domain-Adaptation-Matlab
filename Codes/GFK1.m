
pie05 = load('PIE05.mat');
pie07 = load('PIE07.mat');
pie09 = load('PIE09.mat');
pie27 = load('PIE27.mat');
pie29 = load('PIE29.mat');
Xs = {pie05.fea, pie07.fea, pie09.fea, pie27.fea, pie29.fea};
Xt = {pie05.fea, pie07.fea, pie09.fea, pie27.fea, pie29.fea};
Ys = {pie05.gnd, pie07.gnd, pie09.gnd, pie27.gnd, pie29.gnd};
Yt = {pie05.gnd, pie07.gnd, pie09.gnd, pie27.gnd, pie29.gnd};

dim = 30;
accuracies = cell(numel(Xs), numel(Xt)); % Initialize a cell array

tic
for i = 1:numel(Xs) 
    for j = 1:numel(Xt)
        if i ~= j
            currXs = Xs{i};
            currXt = Xt{j};
            currYs = Ys{i};
            currYt = Yt{j};
            
            accuracies{i, j} = MyGFK(currXs, currYs, currXt, currYt, dim);
            
            %[acc] = JDA(currXs, currXt, currYs, currYt);
        end
    end
end
toc
%{
amazon = load('dslr_SURF_L10.mat');
dslr = load('webcam_SURF_L10.mat');
X_src = amazon.fts;
X_tar = dslr.fts;
Y_src = amazon.labels;
Y_tar = dslr.labels;
dim = 150;

[acc,G,Cls] = MyGFK(X_src,Y_src,X_tar,Y_tar,dim);
%}
function [acc] = MyGFK(X_src,Y_src,X_tar,Y_tar,dim)

    Ps = pca(X_src);
    Pt = pca(X_tar);
    G = GFK_core([Ps,null(Ps')], Pt(:,1:dim));
    [Cls, acc] = my_kernel_knn(G, X_src, Y_src, X_tar, Y_tar);
end


function [prediction,accuracy] = my_kernel_knn(M, Xr, Yr, Xt, Yt)
    dist = repmat(diag(Xr*M*Xr'),1,length(Yt)) ...
        + repmat(diag(Xt*M*Xt')',length(Yr),1)...
        - 2*Xr*M*Xt';
    [~, minIDX] = min(dist);
    prediction = Yr(minIDX);
    accuracy = sum( prediction==Yt ) / length(Yt); 
end

function G = GFK_core(Q,Pt)

    N = size(Q,2); 
    dim = size(Pt,2);

    % compute the principal angles
    QPt = Q' * Pt;
    [V1,V2,V,Gam,Sig] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:));
    V2 = -V2;
    theta = real(acos(diag(Gam))); % theta is real in theory. Imaginary part is due to the computation issue.

    % compute the geodesic flow kernel
    eps = 1e-20;
    B1 = 0.5.*diag(1+sin(2*theta)./2./max(theta,eps));
    B2 = 0.5.*diag((-1+cos(2*theta))./2./max(theta,eps));
    B3 = B2;
    B4 = 0.5.*diag(1-sin(2*theta)./2./max(theta,eps));
    G = Q * [V1, zeros(dim,N-dim); zeros(N-dim,dim), V2] ...
        * [B1,B2,zeros(dim,N-2*dim);B3,B4,zeros(dim,N-2*dim);zeros(N-2*dim,N)]...
        * [V1, zeros(dim,N-dim); zeros(N-dim,dim), V2]' * Q';
end