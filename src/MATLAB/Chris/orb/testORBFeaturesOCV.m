%% Test ORB features
% Tests the OpenCV ORB features

% setup vl_feat toolbox
addpath('..\..\vlfeat-0.9.20\toolbox')
vl_setup

hFig = figure;
%% Step 1: Read Images
original = imresize(imread('1a.jpg'), 2);
new = imresize(imread('1b.jpg'), 2);
subplot(1,3,1)
imshow(original);
subplot(1,3,3)
imshow(new)

%% Step 3: Find Matching Features Between Images
% Detect features in both images.
ptsOriginal  = detectORBFeaturesOCV(original);
ptsNew = detectORBFeaturesOCV(new);

%%
% Extract feature descriptors.
[featuresOriginal_uint8,  validPtsOriginal]  = extractORBFeaturesOCV(original,  ptsOriginal);
[featuresNew_uint8, validPtsNew] = extractORBFeaturesOCV(new, ptsNew);
featuresOriginal = binaryFeatures(featuresOriginal_uint8);
featuresNew = binaryFeatures(featuresNew_uint8);
%%
% Match features by using their descriptors.
indexPairs = matchFeatures(featuresOriginal, featuresNew, 'MatchThreshold', 100.0,'MaxRatio',0.9);

%%
% Retrieve locations of corresponding points for each image.
matchedOriginal  = validPtsOriginal.Location(indexPairs(:,1),:);
matchedNew = validPtsNew.Location(indexPairs(:,2),:);

%%
% Show putative point matches.
% figure;
subplot(1,3,2)
showMatchedFeatures(original,new,matchedOriginal,matchedNew);

figure;
showMatchedFeatures(original,new,matchedOriginal,matchedNew,'montage');
title('Matched ORB features')

figure;
fOriginal= [validPtsOriginal.Location validPtsOriginal.Scale/10 validPtsOriginal.Metric];
fNew= [validPtsNew.Location validPtsNew.Scale/10 validPtsNew.Metric];
subplot(1,2,1)
imshow(original);
vl_plotframe(fOriginal');

subplot(1,2,2)
imshow(new);
vl_plotframe(fNew');
