
% Load Iris dataset
load fisheriris
X = meas;
y = species;

% Convert categorical labels to numerical labels
y_numeric = grp2idx(y);

% Split the data into training and testing sets
rng(42);  % For reproducibility
cv = cvpartition(y_numeric, 'Holdout', 0.2);
trainX = X(training(cv), :);
trainY = y_numeric(training(cv));
testX = X(test(cv), :);
testY = y_numeric(test(cv));

% Train SVM with RBF kernel and OvA approach
SVMModel = fitcecoc(trainX, trainY, 'Learners', templateSVM('KernelFunction', 'RBF'));

% Predict on the test set
predictedY = predict(SVMModel, testX);

% Calculate accuracy
accuracy = sum(predictedY == testY) / numel(testY) * 100;
fprintf('Accuracy: %.2f%%\n', accuracy);

% ... (Visualization code remains the same)


% Plot the SVM decision boundary
figure;
h = plot(SVMModel, 'FillDecisionBoundary', true);
set(h, 'LineWidth', 2);
title('SVM Decision Boundary with RBF Kernel');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Setosa', 'Versicolor', 'Virginica');

% Create confusion matrix
confMat = confusionmat(testY, predictedY);

% Display confusion matrix
figure;
heatmap(confMat, {'Setosa', 'Versicolor', 'Virginica'}, ...
    {'Setosa', 'Versicolor', 'Virginica'}, 1, 'Colormap', 'red', 'ColorbarVisible', 'off');
title('Confusion Matrix');

% Obtain probability estimates for each class
[~, scores] = predict(SVMModel, testX);

% Plot ROC curve
figure;
plotroc(testY, scores);
title('ROC Curve');

% Plot Precision-Recall curve
figure;
plotconfusion(ind2vec(testY'), ind2vec(predictedY'));
title('Precision-Recall Curve');