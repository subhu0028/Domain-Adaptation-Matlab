%{
amazon = load('webcam_SURF_L10.mat');
dslr = load('dslr_SURF_L10.mat');

X_src = amazon.fts;
X_tar = dslr.fts;
Y_src = amazon.labels;
Y_tar = dslr.labels;
%}

pie05 = load('PIE05.mat');
pie07 = load('PIE07.mat');
pie09 = load('PIE09.mat');
pie27 = load('PIE27.mat');
pie29 = load('PIE29.mat');
Xs = {pie05.fea, pie07.fea, pie09.fea, pie27.fea, pie29.fea};
Xt = {pie05.fea, pie07.fea, pie09.fea, pie27.fea, pie29.fea};
Ys = {pie05.gnd, pie07.gnd, pie09.gnd, pie27.gnd, pie29.gnd};
Yt = {pie05.gnd, pie07.gnd, pie09.gnd, pie27.gnd, pie29.gnd};
accuracies = cell(numel(Xs), numel(Xt)); % Initialize a cell array

tic
for i = 1:numel(Xs) 
    for j = 1:numel(Xt)
        if i ~= j
            currXs = Xs{i};
            currXt = Xt{j};
            currYs = Ys{i};
            currYt = Yt{j};
            
            accuracies{i, j} = MyTJM(currXs, currYs, currXt, currYt);
            
            %[acc] = JDA(currXs, currXt, currYs, currYt);
        end
    end
end
toc
%[acc,acc_list,A] = MyTJM(X_src,Y_src,X_tar,Y_tar);

function [acc] = MyTJM(X_src,Y_src,X_tar,Y_tar)

	%% Set options
	lambda = 10;              %% lambda for the regularization
	dim = 20;                    %% dim is the dimension after adaptation, dim <= m
	kernel_type = 'rbf';    %% kernel_type is the kernel name, primal|linear|rbf
	gamma = 0.1;                %% gamma is the bandwidth of rbf kernel
	T = 10;                        %% iteration number
    
	%fprintf('TJM: dim=%d  lambda=%f\n',dim,lambda);

	% Set predefined variables
	X = [X_src',X_tar'];
	X = X*diag(sparse(1./sqrt(sum(X.^2))));
	ns = size(X_src,1);
	nt = size(X_tar,1);
	n = ns+nt;

	% Construct kernel matrix
	K = kernel_tjm(kernel_type,X,[],gamma);

	% Construct centering matrix
	H = eye(n)-1/(n)*ones(n,n);

	% Construct MMD matrix
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
    C = length(unique(Y_src));
	M = e*e' * C;
	
    Cls = [];
	% Transfer Joint Matching: JTM
	G = speye(n);
	acc_list = [];
	for t = 1:T
        %%% Mc [If want to add conditional distribution]
        N = 0;
        if ~isempty(Cls) && length(Cls)==nt
            for c = reshape(unique(Y_src),1,C)
                e = zeros(n,1);
                e(Y_src==c) = 1 / length(find(Y_src==c));
                e(ns+find(Cls==c)) = -1 / length(find(Cls==c));
                e(isinf(e)) = 0;
                N = N + e*e';
            end
        end
        M = M + N;
        
        M = M/norm(M,'fro');
        
	    [A,~] = eigs(K*M*K'+lambda*G,K*H*K',dim,'SM');
	%     [A,~] = eigs(X*M*X'+lambda*G,X*H*X',k,'SM');
	    G(1:ns,1:ns) = diag(sparse(1./(sqrt(sum(A(1:ns,:).^2,2)+eps))));
	    Z = A'*K;
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
	    Zs = Z(:,1:ns)';
	    Zt = Z(:,ns+1:n)';
        
	    knn_model = fitcknn(Zs,Y_src,'NumNeighbors',1);
	    Cls = knn_model.predict(Zt);
	    acc = sum(Cls==Y_tar)/nt;
	    acc_list = [acc_list;acc(1)];
	    
	    %fprintf('[%d]  acc=%f\n',t,full(acc(1)));
    end
    
	fprintf('Algorithm JTM terminated!!!\n\n');
    
    
end


% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'linear','rbf','sam'
%       X:      data matrix (features * samples)
%       gamma:  bandwidth of the RBF/SAM kernel
% Output:
%       K: kernel matrix

function K = kernel_tjm(ker,X,X2,gamma)

switch ker
    case 'linear'
        
        if isempty(X2)
            K = X'*X;
        else
            K = X'*X2;
        end

    case 'rbf'

        n1sq = sum(X.^2,1);
        n1 = size(X,2);

        if isempty(X2)
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
        else
            n2sq = sum(X2.^2,1);
            n2 = size(X2,2);
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
        end
        K = exp(-gamma*D); 

    case 'sam'
            
        if isempty(X2)
            D = X'*X;
        else
            D = X'*X2;
        end
        K = exp(-gamma*acos(D).^2);

    otherwise
        error(['Unsupported kernel ' ker])
end
end