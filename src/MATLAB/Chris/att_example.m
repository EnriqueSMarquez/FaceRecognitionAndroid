%% Test AT&T faces dataset
% This simple example applies HOG and SVM to create a face recognizer
% using the SelfieSecure object

clear

% import AT&T face DB
faceDatabase = imageSet('..\..\..\Data\att_faces','recursive');

% create SelfieSecure object
SelfieSecure_obj = SelfieSecure(faceDatabase);

% partition data into 80% training and 20% testing
SelfieSecure_obj.partitionFaceDatabase([0.8 0.2]);

% train face classifier
SelfieSecure_obj.train();

% test 
[Accuracy,C,order] = SelfieSecure_obj.test