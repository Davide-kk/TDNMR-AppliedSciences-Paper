%%BEST Principal Component Discriminant Analysis (Quadratic)
%[PCcoefs,PCscores_train,~,~,PCexpl] = pca(X_train,'Centered', false);
%PCscores_test = X_test*PCcoefs; (already included in the workspace)
load('PCDA_forthepaper.mat');
mdlDA = fitcdiscr(PCscores_train(:,[3 4 5 8]),Y_train,'DiscrimType','quadratic');
CVmdlDA = crossval(mdlDA,'CVPartition',cDA); % 3 4 5 8 best PCs combo from F.F.S.
[CVpredictDA,scores] = kfoldPredict(CVmdlDA);
figure(1)
confusionchart(Y_train,CVpredictDA);
lossCVDA = kfoldLoss(CVmdlDA);
label = predict(mdlDA,PCscores_test(:,[3 4 5 8]));
figure(2)
confusionchart(Y_test,label)
lossDA = loss(mdlDA,PCscores_test(:,[3 4 5 8]),Y_test);
fprintf('CV error rate = %f',lossCVDA)
fprintf('TEST error rate = %f',lossDA)
%%
[rocX,rocY,thershold,auc] = perfcurve(CVmdlDA.Y,scores(:,1),0);
figure(3)
h = plot(rocX,rocY,'-k','LineWidth',3);
hold on
patch([rocX;1],[rocY;0],[1 0 0],'FaceAlpha',0.2)
plot(5/37,7/11,'ok','MarkerSize',8,'MarkerFaceColor',[1 1 1]);
title("Quadratic Discriminant");
xlabel("False Positive Rate");
ylabel("True Positive Rate");
legend("ROC curve","AUC","Current Model");
grid on
hold off

%%
figure(3)
scatter3(PCscores_train(:,3),PCscores_train(:,4),PCscores_train(:,5),40,Y_train,'filled')
colorbar
xlabel("PC3");
ylabel("PC4");
zlabel("PC5");
%%
timez = Training.Time{1};
timez(1:4) = [];
timez(1074:end) = [];
dataz = Training.Data{1};
dataz(1:4) = [];
dataz(1074:end) = [];

figure(5)
subplot(2,2,1)
yyaxis left
plot(timez,dataz)
ylabel("Intensity (%)");
xlabel("Time (ms)");
title("PC3");
yyaxis right
plot(timez,PCcoefs(:,3))
ylabel("PC loadings");

subplot(2,2,2)
yyaxis left
plot(timez,dataz)
ylabel("Intensity (%)");
xlabel("Time (ms)");
title("PC4");
yyaxis right
plot(timez,PCcoefs(:,4))
ylabel("PC loadings");

subplot(2,2,3)
yyaxis left
plot(timez,dataz)
ylabel("Intensity (%)");
xlabel("Time (ms)");
title("PC5");
yyaxis right
plot(timez,PCcoefs(:,5))
ylabel("PC loadings");

subplot(2,2,4)
yyaxis left
plot(timez,dataz)
ylabel("Intensity (%)");
xlabel("Time (ms)");
title("PC8");
yyaxis right
plot(timez,PCcoefs(:,8))
ylabel("PC loadings");