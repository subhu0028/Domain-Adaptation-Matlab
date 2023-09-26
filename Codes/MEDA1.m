%{
webcam = load('webcam_SURF_L10.mat');
dslr = load('dslr_SURF_L10.mat');
caltech = load('Caltech10_SURF_L10.mat');
amazon = load('amazon_SURF_L10.mat');
%}
webcam = load('webcam_decaf.mat');
dslr = load('dslr_decaf.mat');
caltech = load('caltech_decaf.mat');
amazon = load('amazon_decaf.mat');


Xs = {amazon.feas, caltech.feas, dslr.feas, webcam.feas};
Xt = {amazon.feas, caltech.feas, dslr.feas, webcam.feas};
Ys = {amazon.labels, caltech.labels, dslr.labels, webcam.labels};
Yt = {amazon.labels, caltech.labels, dslr.labels, webcam.labels};



%{
pie05 = load('PIE05.mat');
pie07 = load('PIE07.mat');
pie09 = load('PIE09.mat');
pie27 = load('PIE27.mat');
pie29 = load('PIE29.mat');
Xs = {pie05.fea, pie07.fea, pie09.fea, pie27.fea, pie29.fea};
Xt = {pie05.fea, pie07.fea, pie09.fea, pie27.fea, pie29.fea};
Ys = {pie05.gnd, pie07.gnd, pie09.gnd, pie27.gnd, pie29.gnd};
Yt = {pie05.gnd, pie07.gnd, pie09.gnd, pie27.gnd, pie29.gnd};
%}
accuracies = cell(numel(Xs), numel(Xt)); % Initialize a cell array

tic
for i = 1:numel(Xs) 
    for j = 1:numel(Xt)
        if i ~= j
            currXs = Xs{i};
            currXs = zscore(currXs);
            currXt = Xt{j};
            currXt = zscore(currXt);
            currYs = Ys{i};
            currYt = Yt{j};
            options.d = 20;
            options.rho = 1.0;
            options.p = 10;
            options.lambda = 10.0;
            options.eta = 0.1;
            options.T = 10;
            accuracies{i, j} = MEDA(currXs, currYs, currXt, currYt, options);
            fprintf('%.4f\n', accuracies{i, j});
            %[acc] = JDA(currXs, currXt, currYs, currYt);
        end
    end
end
toc

function [Acc,acc_iter,Beta,Yt_pred] = MEDA(Xs,Ys,Xt,Yt, options)

% Reference:
%% Jindong Wang, Wenjie Feng, Yiqiang Chen, Han Yu, Meiyu Huang, Philip S.
%% Yu. Visual Domain Adaptation with Manifold Embedded Distribution
%% Alignment. ACM Multimedia conference 2018.

%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim
%%% Yt      : Target domain label matrix, m * 1 (only used for testing accuracy)
%%% options : algorithm options:
%%%%% options.d      :  dimension after manifold feature learning (default: 20)
%%%%% options.T      :  number of iteration (default: 10)
%%%%% options.lambda :  lambda in the paper (default: 10)
%%%%% options.eta    :  eta in the paper (default: 0.1)
%%%%% options.rho    :  rho in the paper (default: 1.0)
%%%%% options.base   :  base classifier for soft labels (default: NN)

%% Outputs:
%%%% Acc      :  Final accuracy value
%%%% acc_iter :  Accuracy value list of all iterations, T * 1
%%%% Beta     :  Cofficient matrix
%%%% Yt_pred  :  Prediction labels for target domain

%% Algorithm starts here
    %fprintf('MEDA starts...\n');
    
    %% Load algorithm options
    if ~isfield(options,'p')
        options.p = 10;
    end
    if ~isfield(options,'eta')
        options.eta = 0.1;
    end
    if ~isfield(options,'lambda')
        options.lambda = 1.0;
    end
    if ~isfield(options,'rho')
        options.rho = 1.0;
    end
    if ~isfield(options,'T')
        options.T = 10;
    end
    if ~isfield(options,'d')
        options.d = 20;
    end

    % Manifold feature learning
    [Xs_new,Xt_new,~] = GFK_Map(Xs,Xt,options.d);
    Xs = double(Xs_new');
    Xt = double(Xt_new');

    X = [Xs,Xt];
    n = size(Xs,2);
    m = size(Xt,2);
    C = length(unique(Ys));
    acc_iter = [];
    
    YY = [];
    for c = 1 : C
        YY = [YY,Ys==c];
    end
    YY = [YY;zeros(m,C)];

    %% Data normalization
    X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));

    %% Construct graph Laplacian
    if options.rho > 0
        manifold.k = options.p;
        manifold.Metric = 'Cosine';
        manifold.NeighborMode = 'KNN';
        manifold.WeightMode = 'Cosine';
        W = lapgraph(X',manifold);
        Dw = diag(sparse(sqrt(1 ./ sum(W))));
        L = eye(n + m) - Dw * W * Dw;
    else
        L = 0;
    end

   
    
    % Generate soft labels for the target domain
    knn_model = fitcknn(X(:,1:n)',Ys,'NumNeighbors',1);
    Cls = knn_model.predict(X(:,n + 1:end)');
    

    %YY = [YY;zeros(m,C)];
    
    % Construct kernel
    K = kernel_meda('rbf',X,sqrt(sum(sum(X .^ 2).^0.5)/(n + m)));
    E = diag(sparse([ones(n,1);zeros(m,1)]));
%YY = [];
    for t = 1 : options.T
        
%         if (t>1)
%         Yst=[Ys;Cls];
%         for c = 1 : C
%             YY = [YY,Yst==c];
%         end
%    
%         end
        % Estimate mu
        %mu = estimate_mu(Xs',Ys,Xt',Cls);
        % Construct MMD matrix
        e = [1 / n * ones(n,1); -1 / m * ones(m,1)];
        M = e * e' * length(unique(Ys));
        N = 0;
        for c = reshape(unique(Ys),1,length(unique(Ys)))
            e = zeros(n + m,1);
            e(Ys == c) = 1 / length(find(Ys == c));
            e(n + find(Cls == c)) = -1 / length(find(Cls == c));
            e(isinf(e)) = 0;
            N = N + e * e';
        end
        %M = (1 - mu) * M + mu * N;
        M =  M +  N;
        M = M / norm(M,'fro');

        % Compute coefficients vector Beta
        Beta = ((E + options.lambda * M + options.rho * L) * K + options.eta * speye(n + m,n + m)) \ (E * YY);
        F = K * Beta;
        [~,Cls] = max(F,[],2);
        mu=10;

        %% Compute accuracy
        Acc = numel(find(Cls(n+1:end)==Yt)) / m;
        Cls = Cls(n+1:end);
        acc_iter = [acc_iter;Acc];
        %fprintf('Iteration:[%02d]>>mu=%.2f,Acc=%f\n',t,mu,Acc);
    end
    Yt_pred = Cls;
    %fprintf('MEDA ends!\n');
end

function K = kernel_meda(ker,X,sigma)
    switch ker
        case 'linear'
            K = X' * X;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        
        case 'sam'
            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));
        otherwise
            error(['Unsupported kernel ' ker])
    end
end

function [Xs_new,Xt_new,G] = GFK_Map(Xs,Xt,dim)
    Ps = pca(Xs);
    Pt = pca(Xt);
    G = GFK_core([Ps,null(Ps')], Pt(:,1:dim));
    sq_G = real(G^(0.5));
    Xs_new = (sq_G * Xs')';
    Xt_new = (sq_G * Xt')';
end

function G = GFK_core(Q,Pt)
    % Input: Q = [Ps, null(Ps')], where Ps is the source subspace, column-wise orthonormal
    %        Pt: target subsapce, column-wise orthonormal, D-by-d, d < 0.5*D
    % Output: G = \int_{0}^1 \Phi(t)\Phi(t)' dt

    % ref: Geodesic Flow Kernel for Unsupervised Domain Adaptation.  
    % B. Gong, Y. Shi, F. Sha, and K. Grauman.  
    % Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Providence, RI, June 2012.

    % Contact: Boqing Gong (boqinggo@usc.edu)

    N = size(Q,2); % 
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










% lapgraph
function [W, elapse] = lapgraph(fea,options)
%	Usage:
%	W = graph(fea,options)
%
%	fea: Rows of vectors of data points. Each row is x_i
%   options: Struct value in Matlab. The fields in options that can be set:
%           Metric -  Choices are:
%               'Euclidean' - Will use the Euclidean distance of two data 
%                             points to evaluate the "closeness" between 
%                             them. [Default One]
%               'Cosine'    - Will use the cosine value of two vectors
%                             to evaluate the "closeness" between them.
%                             A popular similarity measure used in
%                             Information Retrieval.
%                  
%           NeighborMode -  Indicates how to construct the graph. Choices
%                           are: [Default 'KNN']
%                'KNN'            -  k = 0
%                                       Complete graph
%                                    k > 0
%                                      Put an edge between two nodes if and
%                                      only if they are among the k nearst
%                                      neighbors of each other. You are
%                                      required to provide the parameter k in
%                                      the options. Default k=5.
%               'Supervised'      -  k = 0
%                                       Put an edge between two nodes if and
%                                       only if they belong to same class. 
%                                    k > 0
%                                       Put an edge between two nodes if
%                                       they belong to same class and they
%                                       are among the k nearst neighbors of
%                                       each other. 
%                                    Default: k=0
%                                   You are required to provide the label
%                                   information gnd in the options.
%                                              
%           WeightMode   -  Indicates how to assign weights for each edge
%                           in the graph. Choices are:
%               'Binary'       - 0-1 weighting. Every edge receiveds weight
%                                of 1. [Default One]
%               'HeatKernel'   - If nodes i and j are connected, put weight
%                                W_ij = exp(-norm(x_i - x_j)/2t^2). This
%                                weight mode can only be used under
%                                'Euclidean' metric and you are required to
%                                provide the parameter t.
%               'Cosine'       - If nodes i and j are connected, put weight
%                                cosine(x_i,x_j). Can only be used under
%                                'Cosine' metric.
%               
%            k         -   The parameter needed under 'KNN' NeighborMode.
%                          Default will be 5.
%            gnd       -   The parameter needed under 'Supervised'
%                          NeighborMode.  Colunm vector of the label
%                          information for each data point.
%            bLDA      -   0 or 1. Only effective under 'Supervised'
%                          NeighborMode. If 1, the graph will be constructed
%                          to make LPP exactly same as LDA. Default will be
%                          0. 
%            t         -   The parameter needed under 'HeatKernel'
%                          WeightMode. Default will be 1
%         bNormalized  -   0 or 1. Only effective under 'Cosine' metric.
%                          Indicates whether the fea are already be
%                          normalized to 1. Default will be 0
%      bSelfConnected  -   0 or 1. Indicates whether W(i,i) == 1. Default 1
%                          if 'Supervised' NeighborMode & bLDA == 1,
%                          bSelfConnected will always be 1. Default 1.
%
%
%    Examples:
%
%       fea = rand(50,15);
%       options = [];
%       options.Metric = 'Euclidean';
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'HeatKernel';
%       options.t = 1;
%       W = constructW(fea,options);
%       
%       
%       fea = rand(50,15);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.Metric = 'Euclidean';
%       options.NeighborMode = 'Supervised';
%       options.gnd = gnd;
%       options.WeightMode = 'HeatKernel';
%       options.t = 1;
%       W = constructW(fea,options);
%       
%       
%       fea = rand(50,15);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.Metric = 'Euclidean';
%       options.NeighborMode = 'Supervised';
%       options.gnd = gnd;
%       options.bLDA = 1;
%       W = constructW(fea,options);      
%       
%
%    For more details about the different ways to construct the W, please
%    refer:
%       Deng Cai, Xiaofei He and Jiawei Han, "Document Clustering Using
%       Locality Preserving Indexing" IEEE TKDE, Dec. 2005.
%    
%
%    Written by Deng Cai (dengcai2 AT cs.uiuc.edu), April/2004, Feb/2006,
%                                             May/2007
% 

if (~exist('options','var'))
   options = [];
else
   if ~isstruct(options) 
       error('parameter error!');
   end
end

%=================================================
if ~isfield(options,'Metric')
    options.Metric = 'Cosine';
end

switch lower(options.Metric)
    case {lower('Euclidean')}
    case {lower('Cosine')}
        if ~isfield(options,'bNormalized')
            options.bNormalized = 0;
        end
    otherwise
        error('Metric does not exist!');
end

%=================================================
if ~isfield(options,'NeighborMode')
    options.NeighborMode = 'KNN';
end

switch lower(options.NeighborMode)
    case {lower('KNN')}  %For simplicity, we include the data point itself in the kNN
        if ~isfield(options,'k')
            options.k = 5;
        end
    case {lower('Supervised')}
        if ~isfield(options,'bLDA')
            options.bLDA = 0;
        end
        if options.bLDA
            options.bSelfConnected = 1;
        end
        if ~isfield(options,'k')
            options.k = 0;
        end
        if ~isfield(options,'gnd')
            error('Label(gnd) should be provided under ''Supervised'' NeighborMode!');
        end
        if ~isempty(fea) && length(options.gnd) ~= size(fea,1)
            error('gnd doesn''t match with fea!');
        end
    otherwise
        error('NeighborMode does not exist!');
end

%=================================================

if ~isfield(options,'WeightMode')
    options.WeightMode = 'Binary';
end

bBinary = 0;
switch lower(options.WeightMode)
    case {lower('Binary')}
        bBinary = 1; 
    case {lower('HeatKernel')}
        if ~strcmpi(options.Metric,'Euclidean')
            warning('''HeatKernel'' WeightMode should be used under ''Euclidean'' Metric!');
            options.Metric = 'Euclidean';
        end
        if ~isfield(options,'t')
            options.t = 1;
        end
    case {lower('Cosine')}
        if ~strcmpi(options.Metric,'Cosine')
            warning('''Cosine'' WeightMode should be used under ''Cosine'' Metric!');
            options.Metric = 'Cosine';
        end
        if ~isfield(options,'bNormalized')
            options.bNormalized = 0;
        end
    otherwise
        error('WeightMode does not exist!');
end

%=================================================

if ~isfield(options,'bSelfConnected')
    options.bSelfConnected = 1;
end

%=================================================
tmp_T = cputime;

if isfield(options,'gnd') 
    nSmp = length(options.gnd);
else
    nSmp = size(fea,1);
end
maxM = 62500000; %500M
BlockSize = floor(maxM/(nSmp*3));


if strcmpi(options.NeighborMode,'Supervised')
    Label = unique(options.gnd);
    nLabel = length(Label);
    if options.bLDA
        G = zeros(nSmp,nSmp);
        for idx=1:nLabel
            classIdx = options.gnd==Label(idx);
            G(classIdx,classIdx) = 1/sum(classIdx);
        end
        W = sparse(G);
        elapse = cputime - tmp_T;
        return;
    end
    
    switch lower(options.WeightMode)
        case {lower('Binary')}
            if options.k > 0
                G = zeros(nSmp*(options.k+1),3);
                idNow = 0;
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = EuDist2(fea(classIdx,:),[],0);
                    [dump idx] = sort(D,2); % sort each row
                    clear D dump;
                    idx = idx(:,1:options.k+1);
                    
                    nSmpClass = length(classIdx)*(options.k+1);
                    G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[options.k+1,1]);
                    G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
                    G(idNow+1:nSmpClass+idNow,3) = 1;
                    idNow = idNow+nSmpClass;
                    clear idx
                end
                G = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
                G = max(G,G');
            else
                G = zeros(nSmp,nSmp);
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    G(classIdx,classIdx) = 1;
                end
            end
            
            if ~options.bSelfConnected
                for i=1:size(G,1)
                    G(i,i) = 0;
                end
            end
            
            W = sparse(G);
        case {lower('HeatKernel')}
            if options.k > 0
                G = zeros(nSmp*(options.k+1),3);
                idNow = 0;
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = EuDist2(fea(classIdx,:),[],0);
                    [dump idx] = sort(D,2); % sort each row
                    clear D;
                    idx = idx(:,1:options.k+1);
                    dump = dump(:,1:options.k+1);
                    dump = exp(-dump/(2*options.t^2));
                    
                    nSmpClass = length(classIdx)*(options.k+1);
                    G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[options.k+1,1]);
                    G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
                    G(idNow+1:nSmpClass+idNow,3) = dump(:);
                    idNow = idNow+nSmpClass;
                    clear dump idx
                end
                G = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
            else
                G = zeros(nSmp,nSmp);
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = EuDist2(fea(classIdx,:),[],0);
                    D = exp(-D/(2*options.t^2));
                    G(classIdx,classIdx) = D;
                end
            end
            
            if ~options.bSelfConnected
                for i=1:size(G,1)
                    G(i,i) = 0;
                end
            end

            W = sparse(max(G,G'));
        case {lower('Cosine')}
            if ~options.bNormalized
                [nSmp, nFea] = size(fea);
                if issparse(fea)
                    fea2 = fea';
                    feaNorm = sum(fea2.^2,1).^.5;
                    for i = 1:nSmp
                        fea2(:,i) = fea2(:,i) ./ max(1e-10,feaNorm(i));
                    end
                    fea = fea2';
                    clear fea2;
                else
                    feaNorm = sum(fea.^2,2).^.5;
                    for i = 1:nSmp
                        fea(i,:) = fea(i,:) ./ max(1e-12,feaNorm(i));
                    end
                end

            end

            if options.k > 0
                G = zeros(nSmp*(options.k+1),3);
                idNow = 0;
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    D = fea(classIdx,:)*fea(classIdx,:)';
                    [dump idx] = sort(-D,2); % sort each row
                    clear D;
                    idx = idx(:,1:options.k+1);
                    dump = -dump(:,1:options.k+1);
                    
                    nSmpClass = length(classIdx)*(options.k+1);
                    G(idNow+1:nSmpClass+idNow,1) = repmat(classIdx,[options.k+1,1]);
                    G(idNow+1:nSmpClass+idNow,2) = classIdx(idx(:));
                    G(idNow+1:nSmpClass+idNow,3) = dump(:);
                    idNow = idNow+nSmpClass;
                    clear dump idx
                end
                G = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
            else
                G = zeros(nSmp,nSmp);
                for i=1:nLabel
                    classIdx = find(options.gnd==Label(i));
                    G(classIdx,classIdx) = fea(classIdx,:)*fea(classIdx,:)';
                end
            end

            if ~options.bSelfConnected
                for i=1:size(G,1)
                    G(i,i) = 0;
                end
            end

            W = sparse(max(G,G'));
        otherwise
            error('WeightMode does not exist!');
    end
    elapse = cputime - tmp_T;
    return;
end


if strcmpi(options.NeighborMode,'KNN') && (options.k > 0)
    if strcmpi(options.Metric,'Euclidean')
        G = zeros(nSmp*(options.k+1),3);
        for i = 1:ceil(nSmp/BlockSize)
            if i == ceil(nSmp/BlockSize)
                smpIdx = (i-1)*BlockSize+1:nSmp;
                dist = EuDist2(fea(smpIdx,:),fea,0);
                dist = full(dist);
                [dump idx] = sort(dist,2); % sort each row
                idx = idx(:,1:options.k+1);
                dump = dump(:,1:options.k+1);
                if ~bBinary
                    dump = exp(-dump/(2*options.t^2));
                end
                
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),2) = idx(:);
                if ~bBinary
                    G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),3) = dump(:);
                else
                    G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),3) = 1;
                end
            else
                smpIdx = (i-1)*BlockSize+1:i*BlockSize;
                dist = EuDist2(fea(smpIdx,:),fea,0);
                dist = full(dist);
                [dump idx] = sort(dist,2); % sort each row
                idx = idx(:,1:options.k+1);
                dump = dump(:,1:options.k+1);
                if ~bBinary
                    dump = exp(-dump/(2*options.t^2));
                end
                
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),2) = idx(:);
                if ~bBinary
                    G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),3) = dump(:);
                else
                    G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),3) = 1;
                end
            end
        end

        W = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
    else
        if ~options.bNormalized
            [nSmp, nFea] = size(fea);
            if issparse(fea)
                fea2 = fea';
                clear fea;
                for i = 1:nSmp
                    fea2(:,i) = fea2(:,i) ./ max(1e-10,sum(fea2(:,i).^2,1).^.5);
                end
                fea = fea2';
                clear fea2;
            else
                feaNorm = sum(fea.^2,2).^.5;
                for i = 1:nSmp
                    fea(i,:) = fea(i,:) ./ max(1e-12,feaNorm(i));
                end
            end
        end
        
        G = zeros(nSmp*(options.k+1),3);
        for i = 1:ceil(nSmp/BlockSize)
            if i == ceil(nSmp/BlockSize)
                smpIdx = (i-1)*BlockSize+1:nSmp;
                dist = fea(smpIdx,:)*fea';
                dist = full(dist);
                [dump idx] = sort(-dist,2); % sort each row
                idx = idx(:,1:options.k+1);
                dump = -dump(:,1:options.k+1);

                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),2) = idx(:);
                G((i-1)*BlockSize*(options.k+1)+1:nSmp*(options.k+1),3) = dump(:);
            else
                smpIdx = (i-1)*BlockSize+1:i*BlockSize;
                dist = fea(smpIdx,:)*fea';
                dist = full(dist);
                [dump idx] = sort(-dist,2); % sort each row
                idx = idx(:,1:options.k+1);
                dump = -dump(:,1:options.k+1);

                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),1) = repmat(smpIdx',[options.k+1,1]);
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),2) = idx(:);
                G((i-1)*BlockSize*(options.k+1)+1:i*BlockSize*(options.k+1),3) = dump(:);
            end
        end

        W = sparse(G(:,1),G(:,2),G(:,3),nSmp,nSmp);
    end
    
    if strcmpi(options.WeightMode,'Binary')
        W(find(W)) = 1;
    end
    
    if isfield(options,'bSemiSupervised') && options.bSemiSupervised
        tmpgnd = options.gnd(options.semiSplit);
        
        Label = unique(tmpgnd);
        nLabel = length(Label);
        G = zeros(sum(options.semiSplit),sum(options.semiSplit));
        for idx=1:nLabel
            classIdx = tmpgnd==Label(idx);
            G(classIdx,classIdx) = 1;
        end
        Wsup = sparse(G);
        if ~isfield(options,'SameCategoryWeight')
            options.SameCategoryWeight = 1;
        end
        W(options.semiSplit,options.semiSplit) = (Wsup>0)*options.SameCategoryWeight;
    end
    
    if ~options.bSelfConnected
        for i=1:size(W,1)
            W(i,i) = 0;
        end
    end

    W = max(W,W');
    
    elapse = cputime - tmp_T;
    return;
end


% strcmpi(options.NeighborMode,'KNN') & (options.k == 0)
% Complete Graph

if strcmpi(options.Metric,'Euclidean')
    W = EuDist2(fea,[],0);
    W = exp(-W/(2*options.t^2));
else
    if ~options.bNormalized
%         feaNorm = sum(fea.^2,2).^.5;
%         fea = fea ./ repmat(max(1e-10,feaNorm),1,size(fea,2));
        [nSmp, nFea] = size(fea);
        if issparse(fea)
            fea2 = fea';
            feaNorm = sum(fea2.^2,1).^.5;
            for i = 1:nSmp
                fea2(:,i) = fea2(:,i) ./ max(1e-10,feaNorm(i));
            end
            fea = fea2';
            clear fea2;
        else
            feaNorm = sum(fea.^2,2).^.5;
            for i = 1:nSmp
                fea(i,:) = fea(i,:) ./ max(1e-12,feaNorm(i));
            end
        end
    end
    
%     W = full(fea*fea');
    W = fea*fea';
end

if ~options.bSelfConnected
    for i=1:size(W,1)
        W(i,i) = 0;
    end
end

W = max(W,W');



elapse = cputime - tmp_T;
end

function D = EuDist2(fea_a,fea_b,bSqrt)
% Euclidean Distance matrix
%   D = EuDist(fea_a,fea_b)
%   fea_a:    nSample_a * nFeature
%   fea_b:    nSample_b * nFeature
%   D:      nSample_a * nSample_a
%       or  nSample_a * nSample_b


if ~exist('bSqrt','var')
    bSqrt = 1;
end


if (~exist('fea_b','var')) | isempty(fea_b)
    [nSmp, nFea] = size(fea_a);

    aa = sum(fea_a.*fea_a,2);
    ab = fea_a*fea_a';
    
    aa = full(aa);
    ab = full(ab);

    if bSqrt
        D = sqrt(repmat(aa, 1, nSmp) + repmat(aa', nSmp, 1) - 2*ab);
        D = real(D);
    else
        D = repmat(aa, 1, nSmp) + repmat(aa', nSmp, 1) - 2*ab;
    end
    
    D = max(D,D');
    D = D - diag(diag(D));
    D = abs(D);
else
    [nSmp_a, nFea] = size(fea_a);
    [nSmp_b, nFea] = size(fea_b);
    
    aa = sum(fea_a.*fea_a,2);
    bb = sum(fea_b.*fea_b,2);
    ab = fea_a*fea_b';

    aa = full(aa);
    bb = full(bb);
    ab = full(ab);

    if bSqrt
        D = sqrt(repmat(aa, 1, nSmp_b) + repmat(bb', nSmp_a, 1) - 2*ab);
        D = real(D);
    else
        D = repmat(aa, 1, nSmp_b) + repmat(bb', nSmp_a, 1) - 2*ab;
    end
    
    D = abs(D);
end

end





function [mu,adist_m,adist_c] = estimate_mu(Xs,Ys,Xt,Yt)
    C = length(unique(Ys));
    list_adist_c = [];
    epsilon = 1e-3;
    for i = 1 : C
        index_i = Ys == i;
        Xsi = Xs(index_i,:);
        index_j = Yt == i;
        Xtj = Xt(index_j,:);
        adist_i = adist(Xsi,Xtj);
        list_adist_c = [list_adist_c;adist_i];
    end
    adist_c = mean(list_adist_c);
    
    adist_m = adist(Xs,Xt);
    mu = adist_c / (adist_c + adist_m);
    if mu > 1    % Theoretically mu <= 1, but calculation may be over 1
        mu = 1;
    elseif mu <= epsilon
        mu = 0;  
    end
end

function dist = adist(Xs,Xt)
    Yss = ones(size(Xs,1),1);
    Ytt = ones(size(Xt,1),1) * 2;
    
    % The results of fitclinear() may vary in a very small range, since Matlab uses SGD to optimize SVM.
    % The fluctuation is very small, ignore it
    model_linear = fitclinear([Xs;Xt],[Yss;Ytt],'learner','svm');
    ypred = model_linear.predict([Xs;Xt]);
    error = mae([Yss;Ytt],ypred);
    dist = 2 * (1 - 2 * error);
end