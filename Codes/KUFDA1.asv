
pie05 = load('PIE05.mat');
pie07 = load('PIE07.mat');
pie09 = load('PIE09.mat');
pie27 = load('PIE27.mat');
pie29 = load('PIE29.mat');
Xs = {pie05.fea, pie07.fea, pie09.fea, pie27.fea, pie29.fea};
Xt = {pie05.fea, pie07.fea, pie09.fea, pie27.fea, pie29.fea};
Ys = {pie05.gnd, pie07.gnd, pie09.gnd, pie27.gnd, pie29.gnd};
Yt = {pie05.gnd, pie07.gnd, pie09.gnd, pie27.gnd, pie29.gnd};
accuracies = cell(numel(Xs), numel(Xt));

for i = 1:length(Xs) 
    for j = 1:length(Xt)
        if i ~= j
            currXs = Xs{i};
            currXt = Xt{j};
            currYs = Ys{i};
            currYt = Yt{j};
            currXs = normr(currXs);
            currXt = normr(currXt);

            currXs = currXs';
            currXt = currXt';
            Cls = knnclassification(currXt',currXs',currYs,1, '2norm');
            Yt0 = Cls;
            
            accuracies{i, j} = JGSA(currXs, currXt, currYs, Yt0, currYt);

            %[Xs, Xt, A, Att] = JGSA(currXs, currXt, currYs, Yt0, currYt);
        end
    end
end

function [acc] = JGSA(Xs, Xt, Ys, Yt0, Yt)

% Joint Geometrical and Statistical Alignment for Visual Domain Adaptation.
% IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
% Jing Zhang, Wanqing Li, Philip Ogunbona.

    alpha = 1;
    mu = 1;
    beta = 0.1;
    gamma = 2;
    ker = 'linear';
    k = 30;
    T = 10;

    m = size(Xs,1);
    ns = size(Xs,2);
    nt = size(Xt,2);

    class = unique(Ys);
    C = length(class);
    if strcmp(ker,'primal')
    
    %--------------------------------------------------------------------------
    % compute LDA
        dim = size(Xs,1);
        meanTotal = mean(Xs,2);

        Sw = zeros(dim, dim);
        Sb = zeros(dim, dim);
        for i=1:C
            Xi = Xs(:,find(Ys==class(i)));
            meanClass = mean(Xi,2);
            Hi = eye(size(Xi,2))-1/(size(Xi,2))*ones(size(Xi,2),size(Xi,2));
            Sw = Sw + Xi*Hi*Xi'; % calculate within-class scatter
            Sb = Sb + size(Xi,2)*(meanClass-meanTotal)*(meanClass-meanTotal)'; % calculate between-class scatter
        end
        P = zeros(2*m,2*m);
        P(1:m,1:m) = Sb;
        Q = Sw;
% from here 
        %constructing graph
        % if options.rho > 0
        manifold.k = graph;
        manifold.Metric = 'Cosine';
        manifold.NeighborMode = 'KNN';
        manifold.WeightMode = 'Cosine';
        X1=[Xs,Xt]';
        W2 = lapgraph(X1,manifold);
  
        Dw = diag(sparse(sqrt(1 ./ sum(W2))));
        L = eye(ns + nt) - Dw * W2 * Dw;
       

        L1=X1'*L*X1;
% till here we added extra or KUFLDA

        for t = 1:T
            % Construct MMD matrix
            [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C);

            Ts = Xs*Ms*Xs';
            Tt = Xt*Mt*Xt';
            Tst = Xs*Mst*Xt';
            Tts = Xt*Mts*Xs';
        
        % Construct centering matrix
            Ht = eye(nt)-1/(nt)*ones(nt,nt);
        
            X = [zeros(m,ns) zeros(m,nt); zeros(m,ns) Xt];    
            H = [zeros(ns,ns) zeros(ns,nt); zeros(nt,ns) Ht];

            %Smax = mu*X*H*X'+beta*P;
            Smax = mu*X*H*X'+beta*P+eta*R;
            %Smin = [Ts+alpha*eye(m)+beta*Q, Tst-alpha*eye(m) ; ...
            %    Tts-alpha*eye(m),  Tt+(alpha+mu)*eye(m)];
            Smin = [options.mmd*Ts+alpha*eye(m)+beta*Q+lambda*L1, options.mmd*Tst-alpha*eye(m)+lambda*L1 ; ...
               options.mmd*Tts-alpha*eye(m)+lambda*L1,  options.mmd*Tt+(alpha+mu)*eye(m)+eta*S+lambda*L1];

            [W,~] = eigs(Smax, Smin+1e-9*eye(2*m), k, 'LM');
            A = W(1:m, :);
            Att = W(m+1:end, :);

            Zs = A'*Xs;
            Zt = Att'*Xt;
        
            if T>1
                Yt0 = knnclassification(Zt',Zs',Ys,1, '2norm');  
                acc = length(find(Yt0==Yt))/length(Yt); 
                %fprintf('acc of iter %d: %0.4f\n',t, full(acc));
            end
        end
    else
    
        Xst = [Xs, Xt];   
        nst = size(Xst,2); 
        [Ks, Kt, Kst] = constructKernel(Xs,Xt,ker,gamma);
   %--------------------------------------------------------------------------
    % compute LDA
        dim = size(Ks,2);
        C = length(class);
        meanTotal = mean(Ks,1);

        Sw = zeros(dim, dim);
        Sb = zeros(dim, dim);
        for i=1:C
            Xi = Ks(find(Ys==class(i)),:);
            meanClass = mean(Xi,1);
            Hi = eye(size(Xi,1))-1/(size(Xi,1))*ones(size(Xi,1),size(Xi,1));
            Sw = Sw + Xi'*Hi*Xi; % calculate within-class scatter
            Sb = Sb + size(Xi,1)*(meanClass-meanTotal)'*(meanClass-meanTotal); % calculate between-class scatter
        end
        P = zeros(2*nst,2*nst);
        P(1:nst,1:nst) = Sb;
        Q = Sw;        

        for t = 1:T

        % Construct MMD matrix
            [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C);
        
            Ts = Ks'*Ms*Ks;
            Tt = Kt'*Mt*Kt;
            Tst = Ks'*Mst*Kt;
            Tts = Kt'*Mts*Ks;

            K = [zeros(ns,nst), zeros(ns,nst); zeros(nt,nst), Kt];
            Smax =  mu*K'*K+beta*P;
        
            Smin = [Ts+alpha*Kst+beta*Q, Tst-alpha*Kst;...
                Tts-alpha*Kst, Tt+mu*Kst+alpha*Kst];
            [W,~] = eigs(Smax, Smin+1e-9*eye(2*nst), k, 'LM');
            W = real(W);

            A = W(1:nst, :);
            Att = W(nst+1:end, :);

            Zs = A'*Ks';
            Zt = Att'*Kt';

            if T>1
                Yt0 = knnclassification(Zt',Zs',Ys,1, '2norm');  
                acc = length(find(Yt0==Yt))/length(Yt); 
                %fprintf('acc of iter %d: %0.4f\n',t, full(acc));
            end
        end
    end


    
    Xs = Zs;
    Xt = Zt;
end


function [Ms, Mt, Mst, Mts] = constructMMD(ns,nt,Ys,Yt0,C)
    e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
    es = 1/ns*ones(ns,1);
    et = -1/nt*ones(nt,1);

    M = e*e'*C;
    Ms = es*es'*C;
    Mt = et*et'*C;
    Mst = es*et'*C;
    Mts = et*es'*C;
    if ~isempty(Yt0) && length(Yt0)==nt
        for c = reshape(unique(Ys),1,C)
            es = zeros(ns,1);
            et = zeros(nt,1);
            es(Ys==c) = 1/length(find(Ys==c));
            et(Yt0==c) = -1/length(find(Yt0==c));
            es(isinf(es)) = 0;
            et(isinf(et)) = 0;
            Ms = Ms + es*es';
            Mt = Mt + et*et';
            Mst = Mst + es*et';
            Mts = Mts + et*es';
        end
    end

    Ms = Ms/norm(M,'fro');
    Mt = Mt/norm(M,'fro');
    Mst = Mst/norm(M,'fro');
    Mts = Mts/norm(M,'fro');
end

function [Ks, Kt, Kst] = constructKernel(Xs,Xt,ker,gamma)

    Xst = [Xs, Xt];   
    ns = size(Xs,2);
    nt = size(Xt,2);
    nst = size(Xst,2); 
    Kst0 = km_kernel(Xst',Xst',ker,gamma);
    Ks0 = km_kernel(Xs',Xst',ker,gamma);
    Kt0 = km_kernel(Xt',Xst',ker,gamma);

    oneNst = ones(nst,nst)/nst;
    oneN=ones(ns,nst)/nst;
    oneMtrN=ones(nt,nst)/nst;
    Ks=Ks0-oneN*Kst0-Ks0*oneNst+oneN*Kst0*oneNst;
    Kt=Kt0-oneMtrN*Kst0-Kt0*oneNst+oneMtrN*Kst0*oneNst;
    Kst=Kst0-oneNst*Kst0-Kst0*oneNst+oneNst*Kst0*oneNst;
end


function result = knnclassification(testsamplesX,samplesX, samplesY, Knn,type)

    % Classify using the Nearest neighbor algorithm
    % Inputs:
    % 	samplesX	   - Train samples
    %	samplesY	   - Train labels
    %   testsamplesX   - Test  samples
    %	Knn		       - Number of nearest neighbors 
    %
    % Outputs
    %	result	- Predicted targets
    if nargin < 5
        type = '2norm';
    end

    L			= length(samplesY);
    Uc          = unique(samplesY);

    if (L < Knn),
    error('You specified more neighbors than there are points.')
    end

    N                   = size(testsamplesX, 1);
    result              = zeros(N,1); 
    switch type
    case '2norm'
        for i = 1:N,
            dist            = sum((samplesX - ones(L,1)*testsamplesX(i,:)).^2,2);
            [m, indices]    = sort(dist);  
            n               = hist(samplesY(indices(1:Knn)), Uc);
            [m, best]       = max(n);
            result(i)        = Uc(best);
        end
    case '1norm'
        for i = 1:N,
            dist            = sum(abs(samplesX - ones(L,1)*testsamplesX(i,:)),2);
            [m, indices]    = sort(dist);   
            n               = hist(samplesY(indices(1:Knn)), Uc);
            [m, best]       = max(n);
            result(i)        = Uc(best);
        end
    case 'match'
        for i = 1:N,
            dist            = sum(samplesX == ones(L,1)*testsamplesX(i,:),2);
            [m, indices]    = sort(dist);   
            n               = hist(samplesY(indices(1:Knn)), Uc);
            [m, best]       = max(n);
            result(i)        = Uc(best);
        end
    otherwise
        error('Unknown measure function');
    end
end


function K = km_kernel(X1,X2,ktype,kpar)

    switch ktype
	    case 'gauss'	% Gaussian kernel
		    sgm = kpar;	% kernel width
		
		    dim1 = size(X1,1);
		    dim2 = size(X2,1);
		
		    norms1 = sum(X1.^2,2);
		    norms2 = sum(X2.^2,2);
		
		    mat1 = repmat(norms1,1,dim2);
		    mat2 = repmat(norms2',dim1,1);
		
		    distmat = mat1 + mat2 - 2*X1*X2';	% full distance matrix
            sgm = sgm / mean(mean(distmat)); % added by jing 24/09/2016, median-distance
		    K = exp(-distmat/(2*sgm^2));
		
	    case 'gauss-diag'	% only diagonal of Gaussian kernel
		    sgm = kpar;	% kernel width
		    K = exp(-sum((X1-X2).^2,2)/(2*sgm^2));
		
	    case 'poly'	% polynomial kernel
% 		p = kpar(1);	% polynome order
% 		c = kpar(2);	% additive constant
            p = kpar; % jing
            c = 1; % jing
		
		    K = (X1*X2' + c).^p;
		
	    case 'linear' % linear kernel
		    K = X1*X2';
		
	    otherwise	% default case
		    error ('unknown kernel type')
    end
end
