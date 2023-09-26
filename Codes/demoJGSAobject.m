clear;close all;
datapath = '/Users/subhangi/Documents/MATLAB/ACM_TML_dataset/Calcat_Def/caltech_decaf.mat';
%addpath('I:\newTML\2018cvpr 2017\LDAPL(single_proposed)\transferlearning-master\code\traditional\TCA')
%addpath('I:\newTML\2018cvpr 2017\LDAPL(single_proposed)\transferlearning-master\code\traditional\JDA')
%addpath('I:\newTML\2018cvpr 2017\LDAPL(single_proposed)\transferlearning-master\code\traditional\GFK')
%addpath('I:\newTML\2018cvpr 2017\LDAPL(single_proposed)\transferlearning-master\code\traditional\CORAL')
% Set algorithm parameters
options.k = 30;             % subspace base dimension
options.ker = 'primal';     % kernel type, default='linear' options: linear, primal, gauss, poly

options.T = 10;             % #iterations, default=10

options.alpha= 1;           % the parameter for subspace divergence ||A-B||
options.mu = 1;             % the parameter for target variance
options.beta = 0.1;         % the parameter for P and Q (source discriminaiton)
options.gamma = 2;          % the parameter for kernel

%% parameter for tca
options.lambda = 0.001;          % the parameter for kernel
options.dim = 30;             % subspace base dimension
options.kernel_type = 'primal';

srcStrSURF12 = {'Caltech10','Caltech10','Caltech10','amazon',   'amazon','amazon','webcam',  'webcam', 'webcam','dslr',    'dslr',   'dslr'};
tgtStrSURF12 = {'amazon',   'webcam',   'dslr',     'Caltech10','webcam','dslr',  'Caltech10','amazon','dslr',  'Caltech10','amazon','webcam'};

srcStrDecaf12 = {'caltech','caltech','caltech','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
tgtStrDecaf12 = {'amazon','webcam','dslr','caltech','webcam','dslr','caltech','amazon','dslr','caltech','amazon','webcam'};


%datafeature = 'SURF12';
 datafeature = 'Decaf12';
tic
results = [];
for iData = 6
% for iData = 9
    if strcmp(datafeature,'SURF12')
        src = char(srcStrSURF12{iData});
        tgt = char(tgtStrSURF12{iData});
        options.data = strcat(src,'-vs-',tgt);
        fprintf('Data=%s \n',options.data);


        % load and preprocess data  
        load([datapath 'GFKdata/' src '_SURF_L10.mat']);
        Xs = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Ys = labels;
        Xs = zscore(Xs);
        Xs = normr(Xs)';

        load([datapath 'GFKdata/' tgt '_SURF_L10.mat']);
        Xt = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Yt = labels;
        Xt = zscore(Xt);
        Xt = normr(Xt)';

    elseif strcmp(datafeature,'Decaf12')

        src = char(srcStrDecaf12{iData});
        tgt = char(tgtStrDecaf12{iData});
        options.data = strcat(src,'-vs-',tgt);
        fprintf('Data=%s \n',options.data);

        % load and preprocess data 
        load(datapath);

        sf = strcmp(ms_data.domain_name,src);
        src_dm = find(sf);
        disp(' ');
        disp('src dm: ');
        src_data = select_dm_data(ms_data, src_dm);
        if isempty(src_data)
            disp(['warning: no src data, domain lbl : ' num2str(src_dm)]);
        end
        tf = strcmp(ms_data.domain_name,tgt);
        tgt_dm = find(tf);
        disp(' ');
        disp('tgt dm: ');
        tgt_data = select_dm_data(ms_data, tgt_dm);
        if isempty(tgt_data)
            disp(['warning: no tgt data, domain lbl : ' num2str(tgt_dm)]);
        end
        
        Xs = src_data.ftr;
        Ys = src_data.lbl;
        Xs = normr(Xs);
        Xs = Xs';

        Xt = tgt_data.ftr;
        Yt = tgt_data.lbl;
        Xt = normr(Xt);
        Xt = Xt';

    end

    %Cls = knnclassify(Xt',Xs',Ys,1); 
    Cls =knnclassification(Xt',Xs',Ys,1);
    acc = length(find(Cls==Yt))/length(Yt); 
    fprintf('acc=%0.4f\n',full(acc));

    Yt0 = Cls;
    [Zs, Zt, A, Att] = JGSA(Xs, Xt, Ys, Yt0, Yt, options);
    %[Zs, Zt, A] = TCA(Xs', Xt',options);
    
    %[Z,acc] = JDA(Xs', Ys,Xt',Yt,options);
    %[acc,Z] = GFK(Xs', Ys,Xt',Yt,options.dim);
    %[Z_new] = CORAL(Xs',Xt');
    
%     Cls = knnclassification(Zt,Zs,Ys,1); 
%     acc = length(find(Cls==Yt))/length(Yt) 
%     results = [results;acc];
%     
%     Z=[Zs;Zt];
      %XX=[Xs';Xt'];
      %Z=XX*Z;
     %Z=[Z_new;Xt'];
    
    %save('CORAL_A_D.mat','Z')
end

toc