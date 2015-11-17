%% Test Fisher Vectors method
% This tests the Fisher Vectors-based Face Recognizer

clear

rng(1);% for reproducibility

% create object
SelfieSecure_obj = SelfieSecure;

% import selfie image DB
% imgDatabase = imageSet('..\..\..\Data\SelfiesDataSet_5faces','recursive');

% folder containing facial image gallery
% faceFolder = '..\..\..\Data\SelfiesFaceDataSet_5faces';
faceFolder = '..\..\..\Data\att_faces';%AT&T faces DB
if exist(faceFolder,'dir')
    % add faceDatabase if facial images already exist
    SelfieSecure_obj.faceDatabase = imageSet(faceFolder,'recursive');
else
    % create facial image database from image database. Writes facial image
    % gallery folders and files to the faceFolder
    faceDatabase = SelfieSecure_obj.createFaceDatabase(imgDatabase, faceFolder);
    SelfieSecure_obj.faceDatabase = faceDatabase;
end

% partition data into training and testing
SelfieSecure_obj.partitionFaceDatabase([0.2 0.8]);

% train face classifier
SelfieSecure_obj.train('fishervectors','fishervectors');

% test
[Accuracy,C,order] = SelfieSecure_obj.test
