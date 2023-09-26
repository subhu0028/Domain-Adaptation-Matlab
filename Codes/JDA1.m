

%{
% Step 2: Create a cell array of tables (your list of DataFrames)
% Replace tbl1, tbl2, tbl3 with your actual tables
tables = {tbl1, tbl2, tbl3};

% Step 3: Iterate through the cell array and calculate accuracies
accuracies = zeros(1, numel(tables));

for i = 1:numel(tables)
    accuracies(i) = calculateAccuracy(tables{i});
end

% The 'accuracies' array now contains the accuracy for each table
disp(accuracies);
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

for i = 1:numel(Xs) 
    for j = 1:numel(Xt)
        if i ~= j
            currXs = Xs{i};
            currXt = Xt{j};
            currYs = Ys{i};
            currYt = Yt{j};
            
            accuracies{i, j} = JDA(currXs, currYs, currXt, currYt);
            
            %[acc] = JDA(currXs, currXt, currYs, currYt);
        end
    end
end

%{
amazon = load('PIE29.mat');
dslr = load('PIE09.mat');

X_src = amazon.fea;
X_tar = dslr.fea;
Y_src = amazon.gnd;
Y_tar = dslr.gnd;

[acc, acc_ite, A] = JDA(X_src,Y_src,X_tar,Y_tar);
%}
function [acc] = JDA(X_src,Y_src,X_tar,Y_tar)

	%% Set options            
	T = 10;                        

	acc_ite = [];
	Y_tar_pseudo = [];
	%% Iteration
	for i = 1 : T
		[Z,A] = JDA_core(X_src,Y_src,X_tar,Y_tar_pseudo);
        %normalization for better classification performance
		Z = Z * diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(X_src,1));
        Zt = Z(:,size(X_src,1)+1:end);

        knn_model = fitcknn(Zs',Y_src,'NumNeighbors',1);
        Y_tar_pseudo = knn_model.predict(Zt');
        acc = length(find(Y_tar_pseudo==Y_tar))/length(Y_tar); 
        %fprintf('JDA+NN=%0.4f\n',acc);
        acc_ite = [acc_ite;acc];
	end

end

function [Z,A] = JDA_core(X_src,Y_src,X_tar,Y_tar_pseudo)
	%% Set options
	lambda = 0.05;              %% lambda for the regularization
	dim = 180;                    %% dim is the dimension after adaptation, dim <= m
	kernel_type = 'rbf';    %% kernel_type is the kernel name, primal|linear|rbf
	gamma = 1;                %% gamma is the bandwidth of rbf kernel

	%% Construct MMD matrix
	X = [X_src',X_tar'];
	X = X*diag(sparse(1./sqrt(sum(X.^2))));
	[m,n] = size(X);
	ns = size(X_src,1);
	nt = size(X_tar,1);
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
	C = length(unique(Y_src));

	%%% M0
	M = e * e' * C;  %multiply C for better normalization

	%%% Mc
	N = 0;
	if ~isempty(Y_tar_pseudo) && length(Y_tar_pseudo)==nt
		for c = reshape(unique(Y_src),1,C)
			e = zeros(n,1);
			e(Y_src==c) = 1 / length(find(Y_src==c));
			e(ns+find(Y_tar_pseudo==c)) = -1 / length(find(Y_tar_pseudo==c));
			e(isinf(e)) = 0;
			N = N + e*e';
		end
	end

	M = M + N;
	M = M / norm(M,'fro');

	%% Centering matrix H
	H = eye(n) - 1/n * ones(n,n);

	%% Calculation
	if strcmp(kernel_type,'primal')
		[A,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',dim,'SM');
    	Z = A'*X;
    else
    	K = kernel_jda(kernel_type,X,[],gamma);
    	[A,~] = eigs(K*M*K'+lambda*eye(n),K*H*K',dim,'SM');
    	Z = A'*K;
	end

end

function K = kernel_jda(ker,X,X2,gamma)

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