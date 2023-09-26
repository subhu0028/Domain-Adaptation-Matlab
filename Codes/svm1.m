% Load the Iris dataset
load fisheriris
X = meas;
y = species;

% Convert species names to numerical labels
y_numeric = grp2idx(y);

% Split the dataset into training and testing sets
rng(42); % Set random seed for reproducibility
[trainIdx, testIdx] = crossvalind('HoldOut', y_numeric, 0.2);

X_train = X(trainIdx, :);
y_train = y_numeric(trainIdx);
X_test = X(testIdx, :);
y_test = y_numeric(testIdx);

% Train the SVM classifier
svm_classifier = fitcsvm(X_train, y_train, 'KernelFunction', 'linear');

% Predict labels for the test data
y_pred = predict(svm_classifier, X_test);

% Calculate the accuracy of the classifier
accuracy = sum(y_pred == y_test) / numel(y_test);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
