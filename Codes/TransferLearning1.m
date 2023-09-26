
alex = alexnet;
layers = alex.Layers;

layers(23) = fullyConnectedLayer(5);
layers(25) = classificationLayer;

allImages = imageDatastore('myImages', 'IncludeSubfolders',true, 'LabelSource','foldernames');
[trainingImages, testImages] = splitEachLabel(allImages, 0.8, "randomized");

opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs',20, 'MiniBatchSize', 64);
myNet = trainNetwork(trainingImages, layers, opts);

pridictedLabels = classify(myNet, testImages);
accuracy = mean(predictedLabels == testImages.Labels)