function [result]=SVM_jiaochajianyan11(data,K_class,K_fold)
% macc,msen,mspe,mppv,sacc,ssen,sspe,sppv,
% This function performs K_fold  cross validation for K_class data
% data is a N*M*K_class 3D matrix where N is number of samples of each
% class and M is the demination of feature

% inputs:
% data: the whole feature data of all samples (cell style)
% K_class: number of classified classes
% K_fold：K_fold　cross validation
% output:
% macc,msen,mspe,mppv are the mean values of ACC,SEN, SPE, PPV
% macc,msen,mspe,mppv are the standrad devition of ACC,SEN, SPE, PPV

for i=1:K_class
    [Ni Mi]=size(data{i}());
    N(i)=Ni;M(i)=Mi;
    meanN(i)=floor(Ni/K_fold+0.01);
    rnN(i)={rand(1,Ni)};
end


for j=1:K_fold
if j==1
    test_wine=[];train_wine=[];test_wine_labels =[];train_wine_labels =[];
    for k=1:K_class
        Rk1=rnN{k}();
        [Rn1,rn]=sort(Rk1);      %对他们进行排序，m是从小到大排序后序列，
    test_wine=[test_wine;data{k}(rn(1:meanN(k)),:)];
    test_wine_labels =[test_wine_labels;k*ones(meanN(k),1)];
    train_wine=[train_wine;data{k}(rn(meanN(k)+1:end),:)];
    train_wine_labels =[train_wine_labels;k*ones(N(k)-meanN(k),1)];
    end
    
elseif j>1 && j<K_fold
    test_wine=[];train_wine=[];test_wine_labels =[];train_wine_labels =[];
    for k=1:K_class
        Rk1=rnN{k}();
        [Rn1,rn]=sort(Rk1);      %对他们进行排序，m是从小到大排序后序列，
    test_wine=[test_wine;data{k}(rn((j-1)*meanN(k)+1:j*meanN(k)),:)];
    test_wine_labels =[test_wine_labels;k*ones(meanN(k),1)];
    train_wine=[train_wine;data{k}(rn(j*meanN(k)+1:end),:)];
    train_wine=[train_wine;data{k}(rn(1:(j-1)*meanN(k)),:)];
    train_wine_labels =[train_wine_labels;k*ones(N(k)-meanN(k),1)];
    end
else
    test_wine=[];train_wine=[];test_wine_labels =[];train_wine_labels =[];
    for k=1:K_class
        Rk1=rnN{k}();
        [Rn1,rn]=sort(Rk1);      %对他们进行排序，m是从小到大排序后序列，
    test_wine=[test_wine;data{k}(rn(N(k)-meanN(k)+1:N(k)),:)];
    test_wine_labels =[test_wine_labels;k*ones(meanN(k),1)];
    train_wine=[train_wine;data{k}(rn(1:N(k)-meanN(k)),:)];
    train_wine_labels =[train_wine_labels;k*ones(N(k)-meanN(k),1)];
    end
end


%% 数据预处理
% 数据预处理,将训练集和测试集归一化到[0,1]区间

[mtrain,ntrain] = size(train_wine);
[mtest,ntest] = size(test_wine);
dataset = [train_wine;test_wine];

% mapminmax为MATLAB自带的归一化函数
[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';

train_wine = dataset_scale(1:mtrain,:);
test_wine = dataset_scale( (mtrain+1):(mtrain+mtest),: );
%% 选择最佳的SVM参数c&g
%% SVM网络训练
% % SVM网络训练
%    model = svmtrain(train_wine_labels, train_wine, '-t 2 -c 2 -g 1');%-t 2调整核函数
% % % 
% % % % % SVM网络预测
%   [predict_label] = svmpredict(test_wine_labels, test_wine, model);%, accuracy

 %predict_label=knnclassify(test_wine,train_wine,train_wine_labels,3);

[predict_label ] = fknn(train_wine,train_wine_labels, ...
	    test_wine_labels, 3, 0, true)
 
% Factor = ClassificationDiscriminant.fit(train_wine, train_wine_labels);
% predict_label = predict(Factor, test_wine);
% [predict_label, Scores] = predict(Factor, test_wine);

% [train_wine,test_wine] = pcaForSVM(train_wine,test_wine,90);
% [bestacc,bestc,bestg] = gaSVMcgForClass(train_wine_labels,train_wine);
% %% 利用最佳的参数进行SVM网络训练
% cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
% model = svmtrain(train_wine_labels,train_wine,cmd);
% 
% %% SVM网络预测
% [predict_label,accuracy,dec_values] = svmpredict(test_wine_labels,test_wine,model);

%   Factor = TreeBagger(50, train_wine, train_wine_labels);
%  [predict_label,Scores] = predict(Factor,  test_wine);
%   predict_label=cell2mat(predict_label);
%  predict_label=str2num(predict_label);
%################################
%  model = initlssvm(train_wine,train_wine_labels,'c',[],[],'RBF_kernel');% RBF_kernel lin_kernel  poly_kernel
% model = tunelssvm(model,'gridsearch','crossvalidatelssvm',{5,'misclass'},'code_OneVsOne');%'gridsearch/simplex'选择优化方式
% model = trainlssvm(model);
% % 使用测试数据对建立的LS-SVM测试
% predict_label = simlssvm(model,test_wine);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sennum=0;
for iii=1:K_class
    sennum=sennum+meanN(iii);
end

acc(j,1) = length(find(predict_label == test_wine_labels))/length(test_wine_labels)*100;
sen(j,1)=(sennum-meanN(1)-length(find(predict_label(meanN(1)+1:sennum)==1)))/(sennum-meanN(1))*100;
spe(j,1)=(length(find(predict_label(1:meanN(1))==1)))/meanN(1)*100;
% ppv(j,1)=(sennum-meanN(1)-length(find(predict_label(meanN(1)+1:sennum)==1)))/((sennum-meanN(1)-length(find(predict_label(meanN(1)+1:sennum)==1)))+meanN(1)-length(find(predict_label(1:meanN(1))==1)))*100;
end
macc={acc};
msen={sen};mspe={spe};%mppv={ppv};
result=[macc,msen,mspe];

