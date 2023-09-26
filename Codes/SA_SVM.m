

amazon = load('webcam_decaf.mat');
dslr = load('amazon_decaf.mat');

X_src = amazon.feas;
Source_Data = NormalizeData(X_src);
X_tar = dslr.feas;
Target_Data = NormalizeData(X_tar);
Source_label = amazon.labels;
Target_label = dslr.labels;

Xs = eig(Source_Data);
Xt = eig(Target_Data);

[acc,y_pred,time_pass] =  SA_svm(Source_Data,Source_label,Target_Data,Target_label,Xs,Xt);

function [acc,y_pred,time_pass] =  SA_svm(Source_Data,Source_label,Target_Data,Target_label,Xs,Xt)

time_start = clock();
A = (Xs*Xs')*(Xt*Xt');
Sim = Source_Data * A *  Target_Data';
[acc,y_pred] = SVM_Accuracy (Source_Data, A,Target_label,Sim,Source_label);
time_end = clock();
accuracy_na_svm = LinAccuracy(Source_Data,Target_Data,Source_label,Target_label)	;
time_pass = etime(time_end,time_start);

end

function Data = NormalizeData(Data)
    Data = Data ./ repmat(sum(Data,2),1,size(Data,2)); 
    Data = zscore(Data,1);  
end


function [res,predicted_label] = SVM_Accuracy (trainset, M,testlabelsref,Sim,trainlabels)
	Sim_Trn = trainset * M *  trainset';
	index = [1:1:size(Sim,1)]';
	Sim = [[1:1:size(Sim,2)]' Sim'];
	Sim_Trn = [index Sim_Trn ];    
	
	C = [0.001 0.01 0.1 1.0 10 100 1000 10000];   
    parfor i = 1 :size(C,2)
		model(i) = libsvmtrain(trainlabels, Sim_Trn, sprintf('-t 4 -c %d -v 2 -q',C(i)));
    end	
	[val indx]=max(model); 
    CVal = C(indx);
	
	model = libsvmtrain(trainlabels, Sim_Trn, sprintf('-t 4 -c %d -q',CVal));
	[predicted_label, accuracy, decision_values] = svmpredict(testlabelsref, Sim, model);
	res = accuracy(1,1);
end


function acc = LinAccuracy(trainset,testset,trainlbl,testlbl)	           
		model = trainSVM_Model(trainset,trainlbl);
        [predicted_label, accuracy, decision_values] = svmpredict(testlbl, testset, model);
        acc = accuracy(1,1);	
end

function svmmodel = trainSVM_Model(trainset,trainlbl)
    C = [0.001 0.01 0.1 1.0 10 100 ];   
    parfor i = 1 :size(C,2)
        model(i) = libsvmtrain(double(trainlbl), sparse(double((trainset))),sprintf('-c %d -q -v 2',C(i) )); 
    end
    [val indx]=max(model); 
    CVal = C(indx);
    svmmodel = libsvmtrain(double(trainlbl), sparse(double((trainset))),sprintf('-c %d -q',CVal));
end