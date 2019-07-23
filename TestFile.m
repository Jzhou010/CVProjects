%% Test file
% 
%% Custom CNN + Custom KNN
% This section uses the custom KNN to extract features from the images, and
% the custom KNN is used for classfication based on the features extracted.

convnet2 = CNNnet;

categories = {'deer','dog','horse','cat','frog', 'airplane', 'ship'};
categories2 = {'deer','dog', 'horse', 'ship', 'frog'};
rootFolder = 'cifar10Test';
testSet = imageDatastore(fullfile(rootFolder, categories2), 'LabelSource', 'foldernames');

[testset, ~] = splitEachLabel(testSet, 50, 'randomize'); 

% Setting up trainning data set
rootFolder = 'cifar10Train';
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
[trainingSet, ~] = splitEachLabel(imds, 500, 'randomize'); 

% Extracting Features using Deep-Neural Network
featureLayer = 'fc_2';
trainingFeatures = activations(convnet2, trainingSet, featureLayer);
testFeatures = activations(convnet2, testset, featureLayer);

PredictedLabel = KNN_Classifer(testFeatures, trainingFeatures, trainingSet.Labels, 2);
FclPredictedLabel = classify(convnet2, testset);

confMat = confusionmat(testset.Labels, PredictedLabel);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))

correct = 0;
score = 0;
for i = 1:size(PredictedLabel,1)
    if (testset.Labels(i) == PredictedLabel(i))
        correct = correct + 1;
    end
    
    score = correct / size(FclPredictedLabel,1);
end
        
disp(score);

correct2 = 0;
score2 = 0;
for i = 1:size(FclPredictedLabel,1)
    if (testset.Labels(i) == FclPredictedLabel(i))
        correct2 = correct2 + 1;
    end
    
    score2 = correct2 / size(FclPredictedLabel,1);
end
        
disp(score2);

%% Built-In KNN
% Classify the images using the built-in KNN function.
classifier = fitcknn(trainingFeatures, trainingSet.Labels);
predictedLabels = predict(classifier, testFeatures);

confMat = confusionmat(testset.Labels, predictedLabels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))

%% Alexnet + Custom KNN
% In this section, instead of using the custom CNN, alexnet is used to
% extract features. 

convnet = alexnet; 

testset2 = testset;
testset2.ReadFcn = @readFunctionTrain;

trainingSet2 = trainingSet;
trainingSet2.ReadFcn = @readFunctionTrain;

featureLayer2 = 'fc7';
trainingFeatures2 = activations(convnet, trainingSet2, featureLayer2);
testFeatures2 = activations(convnet, testset2, featureLayer2);


% Classify the testFeatures
PredictedLabel2 = KNN_Classifer(testFeatures2, trainingFeatures2, trainingSet2.Labels, 3);
FclPredictedLabel2 = classify(convnet, testset2);

confMat = confusionmat(testset2.Labels, PredictedLabel2);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))

correct2 = 0;
score2 = 0;
for i = 1:size(FclPredictedLabel2,1)
    if (testset2.Labels(i) == FclPredictedLabel2(i))
        correct = correct + 1;
    end
    
    score2 = correct2 / size(FclPredictedLabel2,1);
end
        
disp(score2);

%%
% Classify test images using built-in KNN function based on features
% extracted using Alexnet. 
classifier = fitcknn(trainingFeatures2, trainingSet2.Labels);
predictedLabels = predict(classifier, testFeatures2);

confMat = confusionmat(testset2.Labels, predictedLabels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))


