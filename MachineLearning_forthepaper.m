%%
%Trying different algorithms on TC samples, using samples obtained at
%time T0 and T1
load('TCworkspace.mat');
%OUTLIER REMOVER
pattern1 = ["TC1_DX_2_1_T0.001",...
            "TC1_DX_2_2_T0.001",...
            "TC3_DX_2_1_T0.001",...
            "TC3_DX_2_2_T0.001"];
idxOut = contains(AllTCSamples.Sample,pattern1);
AllTCSamples(idxOut,:) = [];
idxT2 = AllTCSamples.Day == 'T2';
AllTCSamples(idxT2,:) = [];

predictors = zeros(1129,height(AllTCSamples));
for k = 1:height(AllTCSamples)
    predictors(:,k) = AllTCSamples.Data{k}; 
end
predictors = predictors';
predictors = normalize(predictors); %center and scale
%%
DatasetDivision = cvpartition(AllTCSamples.FrozenCurdPercent,'HoldOut',0.2);
Training = AllTCSamples(training(DatasetDivision),:);
Test = AllTCSamples(test(DatasetDivision),:);

X_train = predictors(training(DatasetDivision),:);
Y_train = AllTCSamples.FrozenCurdPercent(training(DatasetDivision));

X_test = predictors(test(DatasetDivision),:);
Y_test = AllTCSamples.FrozenCurdPercent(test(DatasetDivision));
%%
%ORIGINAL VARIABLE SELECTION
X_train(:,1:4) = []; %Removing first 2 sec 'cos of instrumental error
X_test(:,1:4) = [];
X_train(:,1074:end) = []; %Removing data over 2034 sec 'cos of large noise
X_test(:,1074:end) = [];
%%
%VARIABLE COMPRESSION WITH PCA
[PCcoefs,PCscores_train,~,~,PCexpl] = pca(X_train,'Centered', false);
PCscores_test = X_test*PCcoefs;
%%
%DISCRIMINANT ANALYSIS
%forward sequential feature selection, PC scores are sequentially used to
%build the model, if the added variable dicreases the loss of the model
%calculated from 6-fold crossvalidation, the variable will be kept in the
%model. This process continues until no further variable decreases
%model's loss. 
opts = statset('Display','iter');
cDA = cvpartition(Y_train,'k',6);
funDA = @(XT,yT,Xt,yt)loss(fitcdiscr(XT,yT),Xt,yt);
[toKeepDA,historyDA] = sequentialfs(funDA,PCscores_train,Y_train,'cv',cDA,'direction','forward','options',opts);
mdlDA = fitcdiscr(PCscores_train(:,toKeepDA),Y_train,'DiscrimType','quadratic');
CVmdlDA = crossval(mdlDA,'KFold',6); % 3 4 5 8 best PCs combo from F.F.S.
CVpredictDA = kfoldPredict(CVmdlDA);
figure(1)
confusionchart(Y_train,CVpredictDA);
lossCVDA = kfoldLoss(CVmdlDA);
label = predict(mdlDA,PCscores_test(:,toKeepDA));
figure(2)
confusionchart(Y_test,label)
lossDA = loss(mdlDA,PCscores_test(:,toKeepDA),Y_test);
%%
%K-MEAN NEAREST NEIGHBOURS (the method to build the model is the same of
%that of DA model
funKNN = @(XT,yT,Xt,yt)loss(fitcknn(XT,yT),Xt,yt);
[toKeepKNN,historyKNN] = sequentialfs(funKNN,PCscores_train,Y_train,'cv',cDA,'direction','forward','options',opts);
mdlKNN = fitcknn(PCscores_train(:,toKeepKNN),Y_train);
CVmdlKNN = crossval(mdlKNN,'KFold',6); % 3 4 5 22 41 best PCs combo from F.F.S.
CVpredictKNN = kfoldPredict(CVmdlKNN);
figure(3)
confusionchart(Y_train,CVpredictKNN);
lossCVKNN = kfoldLoss(CVmdlKNN);
labelKNN = predict(mdlKNN,PCscores_test(:,toKeepKNN));
figure(5)
confusionchart(Y_test,labelKNN)
lossKNN = loss(mdlKNN,PCscores_test(:,toKeepKNN),Y_test);
%%
%SUPPORT VECTOR MACHINES
tsvm = templateSVM('Standardize','off','KernelFunction','gaussian');
funSVM = @(XT,yT,Xt,yt)loss(fitcecoc(XT,yT,'Learners',tsvm),Xt,yt);
[toKeepSVM,historySVM] = sequentialfs(funSVM,PCscores_train,Y_train,'cv',cDA,'direction','forward','options',opts);
%%
% rng default
mdlSVM = fitcecoc(PCscores_train(:,toKeepSVM),Y_train,'Learners',tsvm,'OptimizeHyperparameters',{'BoxConstraint','KernelScale'},...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus','CVPartition',cDA));
%%
svmparameters = templateSVM('Standardize','off','KernelFunction','gaussian','BoxConstraint',803.19,'KernelScale',1.2317);
MDLSVM = fitcecoc(PCscores_train(:,toKeepSVM),Y_train,'Learners',svmparameters);
CVmdlSVM = crossval(MDLSVM,'KFold',6); 
CVpredictSVM = kfoldPredict(CVmdlSVM); 
figure(6)
confusionchart(Y_train,CVpredictSVM);
lossCVSVM = kfoldLoss(CVmdlSVM);
labelSVM = predict(mdlSVM,PCscores_test(:,toKeepSVM));
figure(7)
confusionchart(Y_test,labelSVM)
lossSVM = loss(MDLSVM,PCscores_test(:,toKeepSVM),Y_test);


