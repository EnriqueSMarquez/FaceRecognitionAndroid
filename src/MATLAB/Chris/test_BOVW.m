%% Test Bag of Visual Words method
% This tests the Bag of Visual Words-based Face Recognizer

clear

rng(1);% for reproducibility

% create object
SelfieSecure_obj = SelfieSecure;

% import selfie image DB
% imgDatabase = imageSet('..\..\..\Data\SelfiesDataSet_5faces','recursive');
% imgDatabase = imageSet('..\..\..\Data\SelfiesDataSet','recursive');

% folder containing facial image gallery
% faceFolder = '..\..\..\Data\SelfiesFaceDataSet_5faces';
% faceFolder = '..\..\..\Data\SelfiesFaceDataSet';
faceFolder = '..\..\..\Data\att_faces';% AT&T faces
if exist(faceFolder,'dir')
    % add faceDatabase if facial images already exist
    SelfieSecure_obj.faceDatabase = imageSet(faceFolder,'recursive');
else
    % create facial image database from image database. Writes facial image
    % gallery folders and files to the faceFolder
    faceDatabase = SelfieSecure_obj.createFaceDatabase(imgDatabase, faceFolder);
    SelfieSecure_obj.faceDatabase = faceDatabase;
end

% create SelfieSecure object
SelfieSecure_obj.partitionFaceDatabase([0.2 0.8]);

% train face classifier
SelfieSecure_obj.train('vbow','vbow');

% test
[Accuracy,C,order] = SelfieSecure_obj.test
