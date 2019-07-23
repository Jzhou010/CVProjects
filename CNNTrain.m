% Please note: these are 4 of the 10 categories available
% Feel free to choose which ever you like best!
categories = {'deer','dog','horse','cat','frog','ship', 'truck', 'airplane'...
              'automobile', 'bird'};
NumofClass = size(categories,2);

rootFolder = 'cifar10Train';
imds = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');

layers = [imageInputLayer([32 32 3]);
          convolution2dLayer(5,32,'Padding',1);
          reluLayer();
          maxPooling2dLayer(3,'Stride',2);
          convolution2dLayer(5,64,'Padding',1);
          reluLayer();
          maxPooling2dLayer(3,'Stride',2);
          fullyConnectedLayer(256);
          reluLayer();
          dropoutLayer(0.5);
          fullyConnectedLayer(256);
          reluLayer();
          dropoutLayer(0.5);
          fullyConnectedLayer(NumofClass);
          softmaxLayer();
          classificationLayer()];
      
opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 100, ...
    'Verbose', true);

[CNNnet, info3] = trainNetwork(imds, layers, opts);